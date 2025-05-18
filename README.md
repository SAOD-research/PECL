 ### Progressive Exploration-Conformal Learning for Sparsely Annotated Object Detection in Aerial Images

**[Progressive Exploration-Conformal Learning for Sparsely Annotated Object Detection in Aerial Images(NeurIPS 2024)](https://openreview.net/pdf?id=Jzog9gvOf6)**          

### Introduction
The ability to detect aerial objects with limited annotation is pivotal
to the development of real-world aerial intelligence systems. 
In this work, we focus on a demanding but practical sparsely annotated object detection (SAOD) in aerial images, which encompasses a wider variety of aerial scenes with the same number of annotated objects. 
Although most existing SAOD methods rely on fixed thresholding to filter pseudo-labels for enhancing detector performance, adapting to aerial objects proves challenging due to the imbalanced probabilities/confidences associated with predicted aerial objects.
Specifically, the  pseudo-label exploration can be formulated as a decision-making paradigm by adopting a conformal pseudo-label explorer and a multi-clue selection evaluator. 
The conformal pseudo-label explorer learns an adaptive policy by maximizing the cumulative reward, which can decide how to select these high-quality candidates by leveraging their essential characteristics and inter-instance contextual information.
The multi-clue selection evaluator is designed to evaluate the explorer-guided pseudo-label selections by providing an instructive feedback for policy optimization. 
Finally, the explored pseudo-labels can be adopted to guide the optimization of aerial object detector in a closed-looping progressive fashion.
Comprehensive evaluations on two public datasets demonstrate the superiority of our PECL when compared with other state-of-the-art methods in the sparsely annotated aerial object detection task. 

## Results
<!--
|Model                      |Data           |    Backbone     |    MS  |  Rotate | Lr schd  | box AP | Download|
|:-------------:            |:-------------:| :-------------: | :-----:| :-----: | :-----:  | :----: | :---------------------------------------------------------------------------------------: |
|ReDet                      |DOTA-v1.0       |    ReR50-ReFPN     |   -    |   -    |   1x     |  76.25 |    [cfg](configs/ReDet/ReDet_re50_refpn_1x_dota1.py) [model](https://drive.google.com/file/d/1LCz-Q8PJkr-x9kJk7PcCy37W_cPAdmvO/view?usp=sharing) [log](https://drive.google.com/file/d/1OXgenH6YvtyRUwPH8h9f9p9tBCh60Kln/view?usp=sharing)      |
|ReDet                      |DOTA-v1.0       |    ReR50-ReFPN     |   ✓    |   ✓    |   1x     |  80.10 |    [cfg](configs/ReDet/ReDet_re50_refpn_1x_dota1_ms.py) [model](https://drive.google.com/file/d/1uJb75xTFmQu4db1X8NQKuRNNTrN7TtuA/view?usp=sharing) [log](https://drive.google.com/file/d/1reDaa_ouBfLAZj8Z6wEDsOKxDjeLo0Gt/view?usp=sharing)        |
|ReDet                      |DOTA-v1.5       |    ReR50-ReFPN     |   -    |   -    |   1x     |  66.86 |    [cfg](configs/ReDet/ReDet_re50_refpn_1x_dota15.py) [model](https://drive.google.com/file/d/1AjG3-Db_hmZF1YSKRVnq8j_yuxzualRo/view?usp=sharing) [log](https://drive.google.com/file/d/17dsP9EUbLTV9THkOAA3G3jpmIHHnj83-/view?usp=sharing)        |
|ReDet                      |DOTA-v1.5       |    ReR101-ReFPN     |   -    |   -    |   1x     |  67.62 |    [cfg](configs/ReDet/ReDet_re101_refpn_1x_dota15.py) [model](https://drive.google.com/file/d/1vN4ShOqegn4__QY_hgykota20Qa1mnBQ/view?usp=sharing) [log](https://drive.google.com/file/d/1eKiXI91VudU7rGufdEt526cO8kEm9dAc/view?usp=sharing)        |
|ReDet                      |DOTA-v1.5       |    ReR50-ReFPN     |   ✓    |   ✓    |   1x     |  76.80 |    [cfg](configs/ReDet/ReDet_re50_refpn_1x_dota15_ms.py) [model](https://drive.google.com/file/d/1I1IDmt3juw1sm-CT-zaosVVDldAHYBIO/view?usp=sharing) [log](https://drive.google.com/file/d/1T2Eou26T0mpmP93X_XrFk-AhSicLrgGp/view?usp=sharing)        |
|ReDet                      |HRSC2016        |    ReR50-ReFPN     |   -    |   -    |   3x     |  90.46 |    [cfg](configs/ReDet/ReDet_re50_refpn_3x_hrsc2016.py) [model](https://drive.google.com/file/d/1vTU6OeFD6CX4zkQn7szlgL7Qc_MOZpgC/view?usp=sharing) [log](https://drive.google.com/file/d/1csbm3jop9MGOQt8JaEeBg6TEXOZXY-yo/view?usp=sharing)        |
--> 


## Guideline
All experiments in this paper are based on the **mmdetection/mmrotate** framework, depending on the framework used by the baseline method.

Taking **ReDet** as an example, the installation, training, and testing steps can be found in the official ReDet documentation, which is clear and well-documented.  
You can follow the instructions here: [ReDet GitHub](https://github.com/csuhan/ReDet).

The main difference between the baseline method in this paper and the official ReDet lies in the construction of the sparse annotation dataset, which will be explained in detail below.  

Therefore, you should first run the ReDet code successfully. Then, change the data path in `config/ReDet/ReDet_re50_refpn_1x_dota1.py` to the path of the sparse annotation dataset. This way, you can obtain the corresponding baseline results.

### Prepare sparse annotation DOTA dataset
It is recommended to symlink the dataset root to ReDet/data.

Here, we give an example for single scale data preparation of sparse annotation DOTA-v1.0.

First, make sure your initial data are in the following structure.

```data/dota
├── train
│   ├── images
│   └── labelTxt
├── val
│   ├── images
│   └── labelTxt
└── test
    └── images
```
    
Split the original images and create COCO format json **with different label rates**.

`python DOTA_devkit/prepare_dota1.py --srcpath path_to_dota --dstpath path_to_split_1024`

Then you will get data in the following structure

```dota_1024
├── test1024
│   ├──DOTA_test1024.json
│   └──images
└── trainval1024
     ├──DOTA_trainval1024_0.01.json
     ├──DOTA_trainval1024_0.02.json
     ├──DOTA_trainval1024_0.05.json
     ├──DOTA_trainval1024_0.1.json
     └──images
```

The sparse annotation dataset can be downloaded from the following URL:  
[链接](https://pan.quark.cn/s/8f55ae0f1985)



## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Getting Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage.


## Citation
<!--
BibTeX
@InProceedings{han2021ReDet,
    author    = {Han, Jiaming and Ding, Jian and Xue, Nan and Xia, Gui-Song},
    title     = {ReDet: A Rotation-equivariant Detector for Aerial Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2786-2795}
}
--> 
