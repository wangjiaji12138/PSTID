"""Compatibility entry point; implementations live in the modules below."""

from experiments.ablation import (
    RESULTS_ROOT,
    OUTPUT_MD,
    ABL_LABELS,
    METRICS,
    DATASETS,
    MODEL_NAME,
    identify_abl_type,
    load_results,
    pct,
    main,
)


if __name__ == '__main__':
    main()
