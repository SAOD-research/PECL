# Getting Started

This page provides basic tutorials about the usage of ReDet.
For installation instructions, please see [INSTALL.md](INSTALL.md).


## Prepare DOTA dataset.
It is recommended to symlink the dataset root to `PECL/dataset`.

Split the original DOTA images and create COCO format json. 
```
python DOTA_devkit/prepare_dota1.py --srcpath path_to_dota --dstpath path_to_split_1024 (modify path, modify missing rate)
sudo apt update
sudo apt install libgl1-mesa-glx
apt-get update
apt-get install libglib2.0-dev
```

## Train a model

You can use the following commands to train.

```shell
# single-gpu training
python tools/train.py ${CONFIG_FILE} [--work_dir ${RESULT_FILE}]

# multi-gpu training
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [--work_dir ${RESULT_FILE}]
```

## Test a model
You can use the following commands to test.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}]
```
