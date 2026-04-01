# labelme-to-yolo

### Convert LabelMe polygon annotations to Ultralytics YOLO format for instance segmentation — one command, ready-to-train dataset.

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/labelme-to-yolo.svg)](https://pypi.org/project/labelme-to-yolo/)
[![Downloads](https://static.pepy.tech/personalized-badge/labelme-to-yolo?period=month&units=international_system&left_color=grey&right_color=blue&left_text=PyPi%20Downloads)](https://pepy.tech/project/labelme-to-yolo)
[![Stars](https://img.shields.io/github/stars/Tlaloc-Es/labelme-to-yolo?color=yellow&style=flat)](https://github.com/Tlaloc-Es/labelme-to-yolo/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

> **This project is the active continuation of the archived [labelme2yolov7segmentation](https://github.com/Tlaloc-Es/labelme2yolov7segmentation) repository.** All future development happens here.

______________________________________________________________________

## The Problem

You annotated your dataset with [LabelMe](https://github.com/wkentaro/labelme) — drawing polygons, assigning labels, exporting JSON files. Now you want to train a YOLO model and you discover that YOLO expects a completely different format: normalized coordinates, one `.txt` per image, a specific folder structure, and a `project.yml` config file.

Converting by hand is tedious and error-prone. `labelme-to-yolo` does it in a single command.

______________________________________________________________________

## What it does

- Reads all LabelMe `.json` annotation files from a folder
- Normalizes polygon coordinates to the `[0, 1]` range expected by YOLO
- Copies images and writes `.txt` label files into the YOLO folder structure
- Automatically splits your dataset into **train / val / test** sets
- Generates the `project.yml` configuration file ready to pass to Ultralytics

______________________________________________________________________

## Quickstart

Install from PyPI:

```bash
pip install labelme-to-yolo
```

No install required (via pipx):

```bash
pipx run labelme-to-yolo --source-path /labelme/dataset --output-path /yolo/dataset
```

Run the conversion:

```bash
labelme2yolo --source-path /labelme/dataset --output-path /yolo/dataset
```

______________________________________________________________________

## Usage

```
labelme2yolo --source-path PATH --output-path PATH
```

| Option | Description |
| --------------- | ----------------------------------------------------------------- |
| `--source-path` | Folder containing LabelMe `.json` files and their matching images |
| `--output-path` | Destination folder for the YOLO dataset |

Both relative and absolute paths are supported.

### Expected output

Running:

```bash
labelme2yolo --source-path /labelme/dataset --output-path /yolo/datasets
```

Produces:

```
datasets/
├── images/
│   ├── train/
│   │   ├── img_1.jpg
│   │   └── img_2.jpg
│   ├── val/
│   │   └── img_3.jpg
│   └── test/
│       └── img_4.jpg
├── labels/
│   ├── train/
│   │   ├── img_1.txt
│   │   └── img_2.txt
│   ├── val/
│   │   └── img_3.txt
│   └── test/
│       └── img_4.txt
├── train.txt
├── val.txt
├── test.txt
└── project.yml
```

The generated `project.yml` can be passed directly to Ultralytics YOLO:

```python
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
model.train(data="/yolo/datasets/project.yml", epochs=100)
```

______________________________________________________________________

## Installation from source

```bash
git clone https://github.com/Tlaloc-Es/labelme-to-yolo.git
cd labelme-to-yolo
pip install -e .
```

______________________________________________________________________

## Contributing

Contributions are welcome.

1. Fork the repository
1. Create a branch: `git checkout -b my-feature`
1. Commit your changes following [Conventional Commits](https://www.conventionalcommits.org/)
1. Run the tests: `uv run poe test`
1. Push your branch and open a pull request

______________________________________________________________________

## ⭐ If this saved you time, a star helps others find it

Stars help `labelme-to-yolo` appear when developers search for LabelMe and YOLO conversion tools. It takes 2 seconds.

[⭐ Star on GitHub](https://github.com/Tlaloc-Es/labelme-to-yolo/stargazers)

______________________________________________________________________

## License

MIT. See [LICENSE](LICENSE).

## Donation

If you want to support the project you can make a donation at <https://www.buymeacoffee.com/tlaloc> — thanks in advance.
