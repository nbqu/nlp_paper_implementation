import sys

import pytorch_lightning as pl
from transformers import HfArgumentParser

from hparams import DataTrainingArguments, ModelArguments
from utils.training_args import TrainingArguments
from models.bahdanau import BahdanauSeq2SeqSystem

def main(cfg_file):
    parser = HfArgumentParser((TrainingArguments, DataTrainingArguments, ModelArguments))
    args = parser.parse_json_file(cfg_file)



if __name__ == "__main__":
    main(sys.argv[1])
