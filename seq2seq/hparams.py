from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelArguments:
    rnn_input_size: Optional[int] = field(default=512)
    rnn_hidden_size: Optional[int] = field(default=512)
    rnn_num_layers: Optional[int] = field(default=2)
    rnn_bidirectional: Optional[bool] = field(default=True)
    src_vocab_size: Optional[int] = field(default=None)
    tgt_vocab_size: Optional[int] = field(default=None)

    sos_token: Optional[str] = field(default=None)
    pad_token: Optional[str] = field(default=None)
    sampling_ratio: Optional[str] = field(default=0.5)

    learning_rate: Optional[float] = field(default=0.7)
    lr_decay_at: Optional[int] = field(default=5)

    tgt_tokenizer: Optional[Any] = field(default=None)


@dataclass
class DataTrainingArguments:
    src_lang: Optional[str] = field(default='fr')
    src_tokenizer_path: Optional[str] = field(default=None)
    tgt_lang: Optional[str] = field(default='en')
    tgt_tokenizer_path: Optional[str] = field(default=None)

    batch_size: Optional[int] = field(default=64)
    save_dir: Optional[str] = field(default="./")


@dataclass
class TrainingArguments:
    gradient_clip_val: Optional[int] = field(default=5)
    enable_progress_bar: Optional[bool] = field(default=True)
    max_epochs: Optional[int] = field(default=8)
