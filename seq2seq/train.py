import os
import sys

import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from transformers import (DataCollatorWithPadding, HfArgumentParser,
                          PreTrainedTokenizerFast)

from hparams import DataTrainingArguments, ModelArguments, TrainingArguments
from model import Seq2SeqSystem


def split_sentences_from_dataset(examples, lang):
    sent_list = [example[lang] for example in examples['translation']]
    return {lang: sent_list}


def main(cfg_file):
    pl.seed_everything(42)
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(cfg_file[1])

    raw_dataset = load_dataset("opus_books", "en-fr")
    raw_dataset = raw_dataset["train"].train_test_split(test_size=0.2)
    src_dataset = raw_dataset.map(
        split_sentences_from_dataset,
        batched=True,
        remove_columns=['translation', 'id'],
        fn_kwargs={'lang': data_args.src_lang}
    )
    tgt_dataset = raw_dataset.map(
        split_sentences_from_dataset,
        batched=True,
        remove_columns=['translation', 'id'],
        fn_kwargs={'lang': data_args.tgt_lang}
    )

    src_tokenizer = PreTrainedTokenizerFast.from_pretrained(data_args.src_tokenizer_path)
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(data_args.tgt_tokenizer_path)

    model_args.sos_token = src_tokenizer.bos_token_id
    model_args.pad_token = src_tokenizer.pad_token_id
    model_args.src_vocab_size = src_tokenizer.vocab_size
    model_args.tgt_vocab_size = tgt_tokenizer.vocab_size
    model_args.tgt_tokenizer = tgt_tokenizer  # for bleu decoding

    def tokenize_function(examples, lang):
        if lang == data_args.src_lang:
            model_inputs = src_tokenizer(examples[lang])
        else:
            model_inputs = tgt_tokenizer(examples[lang])

        return model_inputs

    src_tokenized = src_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={'lang': data_args.src_lang},
        remove_columns=[data_args.src_lang]
    )
    tgt_tokenized = tgt_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={'lang': data_args.tgt_lang},
        remove_columns=[data_args.tgt_lang]
    )

    src_collator = DataCollatorWithPadding(src_tokenizer,  padding='longest')
    tgt_collator = DataCollatorWithPadding(tgt_tokenizer,  padding='longest')

    src_dataloader_train = DataLoader(src_tokenized['train'],
                                      batch_size=data_args.batch_size,
                                      collate_fn=src_collator,
                                      num_workers=20)
    tgt_dataloader_train = DataLoader(tgt_tokenized['train'],
                                      batch_size=data_args.batch_size,
                                      collate_fn=tgt_collator,
                                      num_workers=20)

    src_dataloader_valid = DataLoader(src_tokenized['test'],
                                      batch_size=data_args.batch_size,
                                      collate_fn=src_collator,
                                      num_workers=20)
    tgt_dataloader_valid = DataLoader(tgt_tokenized['test'],
                                      batch_size=data_args.batch_size,
                                      collate_fn=tgt_collator,
                                      num_workers=20)

    combined_train = CombinedLoader({'src': src_dataloader_train, 'tgt': tgt_dataloader_train})
    combined_valid = CombinedLoader({'src': src_dataloader_valid, 'tgt': tgt_dataloader_valid})
    model = Seq2SeqSystem(model_args)
    logger = CSVLogger(save_dir=data_args.save_dir, name="seq2seq")
    trainer = Trainer.from_argparse_args(training_args, accelerator='gpu', devices=[0], logger=logger)
    trainer.fit(model, combined_train, combined_valid)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(sys.argv)
