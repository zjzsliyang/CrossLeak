![python](https://img.shields.io/badge/python-v3.7-blue) ![TensorFlow](https://img.shields.io/badge/tensorflow-2.1-orange)

# [Nowhere to Hide: Cross-modal Identity Leakage between Biometrics and Devices](https://arxiv.org/pdf/2001.08211.pdf) - WWW 2020.

## Introduction

This is the prototype and code of the paper above. For ethical consideration, the real biometric data and device IDs of our participants cannot be made available at this momement. We will release the public dataset first and release the 'hashed' real world data once we have done it. 

## Data

Download the simulation data through [Google Drive](https://drive.google.com/drive/folders/19MJp8_KDesW39J8QlqNqDIGGnBF_jh3L?usp=sharing). Put the folders in within `data` or anywhere you specify in the `config.yaml`.

## Dependency

The `requirements.txt` specifies all dependencies. Use the following command to install. `pip3 install -r requirements.txt`

## Run

### Front-End

[README](src/eavesdropping/README.md)

### Back End

`python3 src/association/pipeline.py`

## Ciation

If you find this repository and our data useful, please cite our paper.

```
Stay tuned!
```

