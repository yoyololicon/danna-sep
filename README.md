# Danna-Sep: Danna Sama no Source Separation
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/yoyololicon/Danna-Sep)

The winning model I used in [MDX 2021 Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021/winners).

## Installation

```commandline
python setup.py install
```

## Usage

```
usage: danna_sep [-h] [--outdir OUTDIR] [--fast] infile

positional arguments:
  infile           input audio file

optional arguments:
  -h, --help       show this help message and exit
  --outdir OUTDIR  output directory. Default to current working directory
  --fast           faster inference using only two of the models
```

Given an audio file, the program will split it into 4 stems, which are drums, bass, vocals and other, and store them in the given directory as `.wav` files.

When execute the first time it will download our pre-trained models (around 1 to 2 Gb) to the directory specified by the environment variable `DANNA_CHECKPOINTS`, which by default is `~/danna-sep-checkpoints`.

This process is very time comsuming and require at least 16 Gb of RAM.

## Training

Please refer to our [training repo](https://github.com/yoyololicon/music-demixing-challenge-ismir-2021-entry).

## Web Demo

Try Danna-Sep on [Huggingface Spaces](https://huggingface.co/spaces/yoyololicon/Danna-Sep).

## TODO

- [ ] convert to ONNX format
- [ ] Pack it as standalone app