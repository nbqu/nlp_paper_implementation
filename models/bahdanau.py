import csv
import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import adadelta
from transformers import PreTrainedTokenizerFast

"""
PyTorch Implementation of
["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473).
Highly motivated from [The Annotated Encoder Decoder](https://bastings.github.io/annotated_encoder_decoder/).
"""


class BahdanauEncoder(nn.Module):
    def __init__(self, model_args) -> None:
        super().__init__()
        self.args = model_args
        self.embedding = nn.Embedding(
            num_embeddings=model_args.src_vocab_size,
            embedding_dim=model_args.rnn_input_size,
            padding_idx=model_args.pad_token
        )
        self.dropout = nn.Dropout(p=.1)
        self.rnn = nn.GRU(
            input_size=model_args.rnn_input_size,
            hidden_size=model_args.rnn_hidden_size,
            num_layers=model_args.rnn_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

    def fowrard(self, x):
        x.input_ids = self.dropout(self.embedding(x.input_ids))  # [batch, seq_len, input_size]

        src_packed = pack_padded_sequence(x.input_ids, x.length, batch_first=True)

        # enc_output - [batch, seq_len, 2*hidden_size]
        # enc_hidden - [2*num_layers, batch, hidden_size]
        enc_output, enc_hidden = self.rnn(src_packed)
        enc_output, _ = pad_packed_sequence(enc_output, batch_first=True)

        # rtl_enc_hidden = [num_layers, batch, hidden_size]
        # decoder uses transformed backward hidden state as h_0 of it
        rtl_enc_hidden = enc_hidden[1::2]

        return enc_output, rtl_enc_hidden


class BahdanauAttention(nn.Module):
    def __init__(self, model_args) -> None:
        super().__init__()
        self.args = model_args

        self.query_layer = nn.Linear(model_args.rnn_hidden_size, model_args.rnn_hidden_size)
        self.key_layer = nn.Linear(2*model_args.rnn_hidden_size, model_args.rnn_hidden_size)
        self.energy_layer = nn.Linear(model_args.rnn_hidden_size, 1, bias=False)

        nn.init.normal_(self.query_layer.weight, 0, 0.001)
        nn.init.normal_(self.key_layer.weight, 0, 0.001)
        nn.init.constant_(self.energy_layer, 0)

        self.alphas = None

    def forward(self, query, proj_key, value, src_mask):
        """
        query: previous hidden state of last layer of decoder, [batch, 1, hidden_size]
        proj_key: annotation in the source sentence i.e. output of encoder, [batch, seq_len, hidden_size]
        value: [batch, seq_len, 2*hidden_size]
        src_mask: attention mask from source
        """

        proj_query = self.query_layer(query)  # [batch, 1, hidden_size]
        energy = self.energy_layer(torch.tanh(proj_query + proj_key))  # [batch, seq_len, 1]
        energy = energy.transpose(1, 2)  # [batch, 1, seq_len]

        energy.data.masked_fill_(src_mask == 0, -float('inf'))
        alphas = F.softmax(energy, dim=-1)
        self.alphas = alphas

        context = torch.bmm(energy, value)  # [batch, 1, 2*hidden_size]

        return context, alphas


class BahdanauDecoder(nn.Module):
    def __init__(self, model_args) -> None:
        super().__init__()
        self.model_args = model_args
        self.attn = BahdanauAttention(model_args)
        self.generator = Generator(model_args)
        self.intial_hidden = nn.Linear(model_args.rnn_hidden_size, model_args.rnn_hidden_size)

        self.embedding = nn.Embedding(
            num_embeddings=model_args.tgt_vocab_size,
            embedding_dim=model_args.rnn_input_size,
            padding_idx=model_args.pad_token
        )
        self.dropout = nn.Dropout(p=.1)
        self.rnn = nn.GRU(
            input_size=model_args.rnn_input_size + 2*model_args.rnn_hidden_size,
            hidden_size=model_args.rnn_hidden_size,
            num_layers=model_args.rnn_num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=.1
        )

        self.maxout1 = nn.Linear(3*model_args.rnn_hidden_size+model_args.rnn_input_size, 2*model_args.rnn_hidden_size)
        self.pooling = nn.MaxPool1d(2)

    def forward_step(self, prev_y, prev_hidden, proj_key, value, src_mask):
        """
        prev_y: [batch, 1, input_size]
        prev_hidden: [num_layers, batch, hidden_size]
        """
        def get_maxout(self, query, context, prev_y):
            concatenated = torch.cat((query, context, prev_y), dim=2)  # [batch, 1, 3*hidden_size+input_size]
            t_tilde = self.maxout1(concatenated)
            t = self.pooling(t_tilde)  # [batch, 1, hidden_size]

            return t
        query = prev_hidden[-1].unsqueeze(1)  # [batch, 1, hidden_size]
        # context - [batch, 1, 2*hidden_size]
        context, alphas = self.attn(query, proj_key, value, src_mask)

        rnn_input = torch.cat((prev_y, context), dim=2)  # [batch, 1, input_size+2*hidden_size]
        # dec_output - [batch, 1, hidden_size]
        dec_output, dec_hidden = self.rnn(rnn_input, prev_hidden)
        pre_output = get_maxout(query, context, prev_y)

        return dec_output, dec_hidden, pre_output

    def forward(self, y, enc_output, enc_hidden, sampling_eps):
        batch_max_len = y.input_ids.shape[1]

        # y.input_ids = self.dropout(self.embedding(y.input_ids))  # [batch, seq_len, input_size]

        dec_hidden = torch.tanh(self.intial_hidden(enc_hidden))  # [num_layers, batch, hidden_size]
        proj_key = self.attn.key_layer(enc_output)  # [batch, seq_len, hidden_size]

        dec_outputs = []
        final_outputs = []
        final_predictions = []
        top1 = y.input_ids[:, 0]
        for i in range(batch_max_len):
            prev_y = y.input_ids[:, i] if torch.rand(1) < sampling_eps else top1
            prev_y = self.dropout(self.embedding(prev_y)).unsqueeze(1)
            dec_output, dec_hidden, pre_output = self.forward_step(
                prev_y, dec_hidden, proj_key, enc_hidden, y.attention_mask
            )
            final_output, top1 = self.generator(pre_output)
            final_predictions.append(top1)
            dec_outputs.append(dec_output)
            final_outputs.append(final_output)

        dec_outputs = torch.cat(dec_outputs, dim=1)
        final_outputs = torch.cat(final_outputs, dim=1)
        final_predictions = torch.cat(final_predictions, dim=1).squeeze(-1)

        return dec_outputs, final_outputs, final_predictions


class Generator(nn.Module):
    def __init__(self, model_args) -> None:
        super().__init__()
        self.model_args = model_args
        self.proj = nn.Linear(model_args.rnn_hidden_size, model_args.tgt_vocab_size)

    def forward(self, pre_outputs):
        projected_outputs = self.proj(pre_outputs)
        probs = F.log_softmax(projected_outputs, dim=-1)
        top1 = torch.argmax(probs, dim=-1)
        return probs, top1


class BahdanauSeq2Seq(nn.Module):
    def __init__(self, model_args) -> None:
        super().__init__()
        self.encoder = BahdanauEncoder(model_args)
        self.decoder = BahdanauDecoder(model_args)

    def forward(self, x, y, sampling_eps):
        enc_output, enc_hidden = self.encoder(x)
        dec_outputs, final_outputs = self.decoder(y, enc_output, enc_hidden, sampling_eps)
        return final_outputs


class BahdanauSeq2SeqSystem(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = BahdanauSeq2Seq(self.hparams)
        self.tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.hparams.tgt_tokenizer_path)
        self.model.apply(self.init_weights)

    def configure_optimizers(self):
        optimizer = adadelta(self.parameters(), lr=self.model_args.learning_rate, rho=0.95)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        src, tgt = batch['src'], batch['tgt']
        dec_outputs, final_outputs, _ = self.model(src, tgt, self.hparams.sampling_eps)
        loss = F.cross_entropy(
            final_outputs.contiguous().view(-1, self.hparams.tgt_vocab_size), tgt.contiguous().view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch['src'], batch['tgt']
        dec_outputs, final_outputs, final_predictions = self.model(src, tgt, self.hparams.sampling_eps)
        loss = F.cross_entropy(
            final_outputs.contiguous().view(-1, self.hparams.tgt_vocab_size), tgt.contiguous().view(-1))
        return {'loss': loss, 'predictions': final_predictions, 'targets': batch[self.hparams.tgt_lang]}

    def validation_epoch_end(self, outputs) -> None:
        all_predictions = [self.tgt_tokenizer.batch_decode(batch['predictions']) for batch in outputs]
        all_targets = [batch['targets'] for batch in outputs]
        with open(os.path.join(self.logger.log_dir, f'val_{self.current_epoch}.csv'), 'w') as fp:
            writer = csv.DictWriter(fp, fieldnames=['prediction', 'targets'])
            writer.writeheader()
            for pred_batch, target_batch in zip(all_predictions, all_targets):
                for pred, target in zip(pred_batch, target_batch):
                    writer.writerow({'prediction': pred, 'target': target})

    def on_train_batch_end(self, outputs, batch, batch_idx, unused) -> None:
        total_steps = self.trainer.estimated_stepping_batches
        current_step = self.trainer.global_step
        self.hparams.sampling_eps = (total_steps - current_step) / total_steps
        return super().on_train_batch_end(outputs, batch, batch_idx, unused)

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, 0, 0.01)
