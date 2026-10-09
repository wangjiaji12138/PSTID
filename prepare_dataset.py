"""Compatibility entry point; implementations live in the modules below."""

from data_pipeline.npz import (
    DATASET_CONFIG,
    parse_date,
    load_raw_data,
    filter_by_date_range,
    create_sliding_windows,
    save_all_dataset,
    save_adjacency,
    process_dataset,
    main,
)


if __name__ == '__main__':
    main()
