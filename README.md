# NLDL 2025

This repository contains scripts for running machine learning models with 10x cross-validation, and 100x averaged independent dataset runs. Results are stored in JSON format, including metrics like accuracy, balanced accuracy, F1 score, MCC, recall, and precision.

## Files

- **`driver.py`**: Runs `main.py` in a loop for cross-validation and dataset runs, initializes model and dataset names, and compiles results into a JSON file for 100 epochs across all runs in the format (number of runs x number of epochs x number of metrics).
- **`main.py`**: Sets hyperparameters and runs the chosen model, outputting performance metrics after each epoch.
- **`models.py`**: Defines model architectures used by `main.py`.
- **`utils_aug.py`**: Contains dataloaders, data augmentations, tokenizers, training/evaluation loops, and metric calculations.
- **`split.py`**: Splits the dataset into 10 nearly equal datasets for cross-validation
- **`osfp.json`**: Dataset curated from [Simeon et al.](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0185-8)
- **`fpbase.json`**: Dataset curated from [Lambert et al.](https://www.nature.com/articles/s41592-019-0352-8)

## Usage

1. Set up the model and dataset in `driver.py`.
2. Run `driver.py` to execute cross-validation and dataset runs.
3. View results in the generated JSON file.
