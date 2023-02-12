[![Python 3.10.9](https://img.shields.io/badge/python-3.10.9-blue.svg)](https://www.python.org/downloads/release/python-3109/)


# MATRYOSHKA - Russian Visual-Semantic Clothing Dataset

# Develop
## Parse raw data
```shell
python ./scripts/parse_raw_dataset.py --source_path ./matryoshka_raw
```
Parse, archive, delete result folder
```shell
python ./scripts/parse_raw_dataset.py --source_path ./matryoshka_raw --archive_result_path . --remove_result_folder
```
Full args info:
```shell
python ./scripts/parse_raw_dataset.py --help
```