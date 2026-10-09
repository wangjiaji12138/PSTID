"""Compatibility entry point; implementations live in the modules below."""

from training.config import (
    create_parser,
)
from training.reproducibility import (
    set_seed,
)
from data_pipeline.loaders import (
    build_dataloaders,
)
from data_pipeline.normalization import (
    compute_scaler_stats,
    normalize_input,
)
from training.checkpoint import (
    save_checkpoint,
    save_results,
)
from training.engine import (
    train_epoch,
    evaluate,
    train_single_model,
)
from training.runner import (
    MODEL_LIST,
    build_model,
    main,
)


if __name__ == '__main__':
    main()
