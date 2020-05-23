![python](https://img.shields.io/badge/python-v3.7-blue) ![TensorFlow](https://img.shields.io/badge/tensorflow-2.1-orange)

# [Nowhere to Hide: Cross-modal Identity Leakage between Biometrics and Devices](https://arxiv.org/pdf/2001.08211.pdf) - WWW 2020.

## Introduction

This is the reproducible code of the paper above. For ethical consideration, the collected biometric data and device IDs of our participants cannot be made available at this momement. We will release the simulated public dataset first and release the when the real-world collection when it is sufficiently 'hashed' to pass several ethics checks. 

## Public Data

Download the pre-baked data from [Google Drive](https://drive.google.com/drive/folders/19MJp8_KDesW39J8QlqNqDIGGnBF_jh3L?usp=sharing) to the directory `data` or anywhere you specify in the `config.yaml`.

## Dependency

The `requirements.txt` specifies all dependencies. Use the following command to install. `pip3 install -r requirements.txt`

## Run

### Front-End

Check out how to set up the multi-modal sensing frond-end in this [README](src/eavesdropping/README.md)

### Back-End

`python3 src/association/pipeline.py`

## Ciation

If you find this repository useful, please cite our paper.

```
@inproceedings{lu2020nowhere,
  title={{Nowhere to Hide: Cross-modal Identity Leakage between Biometrics and Devices}},
  author={Lu, Chris Xiaoxuan and Li Yang, Xiangli Yuanbo and Li Zhengxiong},
  booktitle={The Web Conference (WWW)},
  year={2020}
}
```

