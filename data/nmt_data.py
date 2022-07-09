import pytorch_lightning as pl
from datasets import load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast


class OpusBooksDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.batch_size = self.hparams.batch_size

    def setup(self, stage: str) -> None:
        self.src_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.hparams.src_tokenizer_path)
        self.tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.hparams.tgt_tokenizer_path)

        self.hparams.sos_token = self.src_tokenizer.bos_token_id
        self.hparams.pad_token = self.src_tokenizer.pad_token_id
        self.hparams.src_vocab_size = self.src_tokenizer.vocab_size
        self.hparams.tgt_vocab_size = self.tgt_tokenizer.vocab_size

        raw_dataset = load_dataset("opus_books", "en-fr", split='train')
        encoded_dataset = raw_dataset.map(self.tokenize_function, batched=True)
        encoded_dataset = encoded_dataset.train_test_split(test_size=0.2)

        self.src_collator = DataCollatorWithPadding(self.src_tokenizer,  padding='longest')
        self.tgt_collator = DataCollatorWithPadding(self.tgt_tokenizer,  padding='longest')


    def tokenize_function(self, examples):
        model_inputs = {}
        source = [example[self.hparams.src_lang] for example in examples['translation']]
        target = [example[self.hparams.tgt_lang] for example in examples['translation']]

        model_inputs[self.hparams.src_lang] = self.src_tokenizer(source, return_length=True)
        model_inputs[self.hparams.tgt_lang] = self.tgt_tokenizer(target, return_length=True)

        return model_inputs
