from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.plugins import PLUGIN_INPUT
from pytorch_lightning.profiler import Profiler
from pytorch_lightning.strategies import Strategy


@dataclass
class TrainingArguments:
    """
    for detail description, see Docs.
    """
    logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = field(default=True),
    checkpoint_callback: Optional[bool] = field(default=None),
    enable_checkpointing: bool = field(default=True),
    callbacks: Optional[Union[List[Callback], Callback]] = field(default=None),
    default_root_dir: Optional[str] = field(default=None),
    gradient_clip_val: Optional[Union[int, float]] = field(default=None),
    gradient_clip_algorithm: Optional[str] = field(default=None),
    process_position: int = 0,
    num_nodes: int = 1,
    num_processes: Optional[int] = field(default=None),
    devices: Optional[Union[List[int], str, int]] = field(default=None),
    gpus: Optional[Union[List[int], str, int]] = field(default=None),
    auto_select_gpus: bool = False,
    tpu_cores: Optional[Union[List[int], str, int]] = field(default=None),
    ipus: Optional[int] = field(default=None),
    enable_progress_bar: bool = field(default=True),
    overfit_batches: Union[int, float] = 0.0,
    track_grad_norm: Union[int, float, str] = -1,
    check_val_every_n_epoch: int = 1,
    fast_dev_run: Union[int, bool] = False,
    accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = field(default=None),
    max_epochs: Optional[int] = field(default=None),
    min_epochs: Optional[int] = field(default=None),
    max_steps: int = -1,
    min_steps: Optional[int] = field(default=None),
    max_time: Optional[Union[str, timedelta, Dict[str, int]]] = field(default=None),
    limit_train_batches: Optional[Union[int, float]] = field(default=None),
    limit_val_batches: Optional[Union[int, float]] = field(default=None),
    limit_test_batches: Optional[Union[int, float]] = field(default=None),
    limit_predict_batches: Optional[Union[int, float]] = field(default=None),
    val_check_interval: Optional[Union[int, float]] = field(default=None),
    flush_logs_every_n_steps: Optional[int] = field(default=None),
    log_every_n_steps: int = 50,
    accelerator: Optional[Union[str, Accelerator]] = field(default=None),
    strategy: Optional[Union[str, Strategy]] = field(default=None),
    sync_batchnorm: bool = False,
    precision: Union[int, str] = 32,
    enable_model_summary: bool = field(default=True),
    weights_summary: Optional[str] = "top",
    num_sanity_val_steps: int = 2,
    resume_from_checkpoint: Optional[Union[Path, str]] = field(default=None),
    profiler: Optional[Union[Profiler, str]] = field(default=None),
    benchmark: Optional[bool] = field(default=None),
    deterministic: Optional[bool] = field(default=None),
    reload_dataloaders_every_n_epochs: int = 0,
    auto_lr_find: Union[bool, str] = False,
    replace_sampler_ddp: bool = field(default=True),
    detect_anomaly: bool = False,
    auto_scale_batch_size: Union[str, bool] = False,
    prepare_data_per_node: Optional[bool] = field(default=None),
    plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = field(default=None),
    amp_backend: str = "native",
    amp_level: Optional[str] = field(default=None),
    move_metrics_to_cpu: bool = False,
    multiple_trainloader_mode: str = "max_size_cycle",
    stochastic_weight_avg: bool = False,
    terminate_on_nan: Optional[bool] = field(default=None),
