from typing import Any

import pytorch_lightning as pl
import torch
from datasets import load_metric
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler


class Seq2SeqEncoder(nn.Module):
    def __init__(self, model_args) -> None:
        super(Seq2SeqEncoder, self).__init__()

        self.args = model_args
        self.embedding = nn.Embedding(
            num_embeddings=model_args.src_vocab_size,
            embedding_dim=model_args.rnn_input_size,
            padding_idx=model_args.pad_token
        )
        self.rnn = nn.LSTM(
            input_size=model_args.rnn_input_size,
            hidden_size=model_args.rnn_hidden_size,
            num_layers=model_args.rnn_num_layers,
            bidirectional=model_args.rnn_bidirectional,
            batch_first=True
        )

    def forward(self, x):
        # x : [batch_size, seq_len, rnn_input_size]
        x = self.embedding(x)
        # rnn_output : [batch_size, seq_len, 2*rnn_hidden_size]
        # hidden, cell : [4, batch_size, rnn_hidden_size]
        rnn_output, (hidden, cell) = self.rnn(x)
        return rnn_output, hidden, cell


class Seq2SeqDecoder(nn.Module):
    def __init__(self, model_args) -> None:
        super(Seq2SeqDecoder, self).__init__()
        self.args = model_args
        self.embedding = nn.Embedding(
            num_embeddings=model_args.src_vocab_size,
            embedding_dim=model_args.rnn_input_size,
            padding_idx=model_args.pad_token
        )
        self.rnn = nn.LSTM(
            input_size=model_args.rnn_input_size,
            hidden_size=model_args.rnn_hidden_size,
            num_layers=model_args.rnn_num_layers,
            bidirectional=model_args.rnn_bidirectional,
            batch_first=True
        )
        in_features = model_args.rnn_hidden_size * 2 if model_args.rnn_bidirectional else model_args.rnn_num_layers
        self.classifier = nn.Linear(in_features, model_args.tgt_vocab_size)

    def forward(self, x, h_0, c_0):
        x = x.unsqueeze(1)

        # x : [batch_size, 1, rnn_input_size]
        x = self.embedding(x)

        # rnn_output : [batch_size, 1, 2*rnn_hidden_size]
        rnn_output, (hidden, cell) = self.rnn(x, (h_0, c_0))

        # output : [batch_size, tgt_vocab_size]
        output = self.classifier(rnn_output.squeeze(1))
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, device, model_args) -> None:
        super(Seq2Seq, self).__init__()
        self.args = model_args
        self.device = device
        self.encoder = Seq2SeqEncoder(model_args)
        self.decoder = Seq2SeqDecoder(model_args)

    def forward(self, src, tgt):
        # hidden, cell : [4, batch_size, rnn_hidden_size]
        _, enc_hid, enc_cell = self.encoder(src)
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.args.tgt_vocab_size

        outputs = torch.zeros((batch_size, tgt_len, tgt_vocab_size), device=tgt.device)
        dec_input = tgt[:, 0]
        dec_hid, dec_cell = enc_hid, enc_cell

        for timestep in range(1, tgt.shape[1]):
            output, dec_hid, dec_cell = self.decoder(dec_input, dec_hid, dec_cell)
            outputs[:, timestep, :] = output
            top1 = output.argmax(-1)
            if torch.rand(1).item() < self.args.sampling_ratio:
                dec_input = tgt[:, timestep]
            else:
                dec_input = top1

        return outputs


class Seq2SeqSystem(pl.LightningModule):
    def __init__(self, model_args) -> None:
        super().__init__()
        self.model_args = model_args
        self.model = Seq2Seq(self.device, model_args)
        self.metric = load_metric("bleu")

        self.model.apply(self.init_weights)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.model_args.learning_rate)
        scheduler = lr_scheduler.ConstantLR(
            optimizer, factor=.5, total_iters=self.model_args.lr_decay_at
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        src, tgt = batch['src'].input_ids, batch['tgt'].input_ids
        outputs = self.model(src, tgt)
        outputs = outputs.view(-1, self.model_args.tgt_vocab_size)
        tgt = tgt.view(-1)
        loss = F.cross_entropy(outputs, tgt)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch['src'].input_ids, batch['tgt'].input_ids
        outputs = self.model(src, tgt)
        val_loss = F.cross_entropy(outputs.view(-1, self.model_args.tgt_vocab_size), tgt.view(-1))

        output_tokens = self.convert_ids_to_tokens(outputs.argmax(-1))
        tgt_tokens = self.convert_ids_to_tokens(tgt, is_gold=True)
        self.metric.add_batch(predictions=output_tokens, references=tgt_tokens)
        self.log(f'val_loss {self.current_epoch}', val_loss.item())

        return val_loss

    def validation_epoch_end(self, outputs):
        self.log(f"bleu {self.current_epoch}", self.metric.compute()['bleu'])
        return super().validation_epoch_end(outputs)

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def convert_ids_to_tokens(self, tensor, is_gold=False):
        li = tensor.tolist()
        if not is_gold:
            return [self.model_args.tgt_tokenizer.convert_ids_to_tokens(
                example, skip_special_tokens=True
                ) for example in li]
        else:
            return [[self.model_args.tgt_tokenizer.convert_ids_to_tokens(
                example, skip_special_tokens=True
                )] for example in li]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        src, tgt = batch['src'].input_ids, batch['tgt'].input_ids
        outputs = self.model(src, tgt)

        final_output = self.decode(outputs.argmax(-1), skip_special_tokens=True)
        final_gold = self.decode(tgt, skip_special_tokens=True)
        return {'pred': final_output, 'gold': final_gold}
