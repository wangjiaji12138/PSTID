"""Compatibility entry point; implementations live in the modules below."""

from data_pipeline.grid import (
    DataProcessor,
    GridSplitter,
    main,
)


if __name__ == '__main__':
    main()
