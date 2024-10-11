### Progressive Exploration-Conformal Learning for Sparsely Annotated Object Detection in Aerial Images

[ReDet: A Rotation-equivariant Detector for Aerial Object Detection (CVPR2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Han_ReDet_A_Rotation-Equivariant_Detector_for_Aerial_Object_Detection_CVPR_2021_paper.pdf)**,            

The repo is based on [AerialDetection](https://github.com/dingjiansw101/AerialDetection) and [mmdetection](https://github.com/open-mmlab/mmdetection).
[AerialDetection](https://github.com/dingjiansw101/AerialDetection) is a powerful framework for object detection in aerial images, which contains a lot of useful algorithms and tools.

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

## Benchmark and model zoo

* **ImageNet pretrain**

We pretrain our ReResNet on the ImageNet-1K. Related codes can be found at the [ReDet_mmcls](https://github.com/csuhan/ReDet/tree/ReDet_mmcls) branch. 
Here we provide our pretrained ReResNet-50 model for convenience. 
If you want to train and use ReResNet in your own project, please check out [ReDet_mmcls](https://github.com/csuhan/ReDet/tree/ReDet_mmcls) for the installation and basic usage.


|         Model                                               |Group      | Top-1 (%) | Top-5 (%) | Download |
|:-----------------------------------------------------------:|:---------:|:---------:|:---------:|:--------:|
| [ReR50](https://github.com/csuhan/ReDet/blob/ReDet_mmcls/configs/re_resnet/re_resnet50_c8_batch256.py) |C<sub>8</sub>| 71.20     | 90.28     | [model](https://drive.google.com/file/d/1FshfREfLZaNl5FcaKrH0lxFyZt50Uyu2/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1VLW8YbU1kGpqd4hfvI9UItbCOprzo-v4/view?usp=sharing)|
| [ReR101](https://github.com/csuhan/ReDet/blob/ReDet_mmcls/configs/re_resnet/re_resnet101_c8_batch256.py) |C<sub>8</sub>| 74.92     | 92.22     | [model](https://drive.google.com/file/d/1GmJzzHRgp5SvmGa6uj6n4GpCuYRT5RE9/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1w1KGCzYFPIJjjVOR2FOGgytYu4oCrjAM/view?usp=sharing)|


* **Object Detection**

|Model                      |Data           |    Backbone     |    MS  |  Rotate | Lr schd  | box AP | Download|
|:-------------:            |:-------------:| :-------------: | :-----:| :-----: | :-----:  | :----: | :---------------------------------------------------------------------------------------: |
|ReDet                      |DOTA-v1.0       |    ReR50-ReFPN     |   -    |   -    |   1x     |  76.25 |    [cfg](configs/ReDet/ReDet_re50_refpn_1x_dota1.py) [model](https://drive.google.com/file/d/1LCz-Q8PJkr-x9kJk7PcCy37W_cPAdmvO/view?usp=sharing) [log](https://drive.google.com/file/d/1OXgenH6YvtyRUwPH8h9f9p9tBCh60Kln/view?usp=sharing)      |
|ReDet                      |DOTA-v1.0       |    ReR50-ReFPN     |   ✓    |   ✓    |   1x     |  80.10 |    [cfg](configs/ReDet/ReDet_re50_refpn_1x_dota1_ms.py) [model](https://drive.google.com/file/d/1uJb75xTFmQu4db1X8NQKuRNNTrN7TtuA/view?usp=sharing) [log](https://drive.google.com/file/d/1reDaa_ouBfLAZj8Z6wEDsOKxDjeLo0Gt/view?usp=sharing)        |
|ReDet                      |DOTA-v1.5       |    ReR50-ReFPN     |   -    |   -    |   1x     |  66.86 |    [cfg](configs/ReDet/ReDet_re50_refpn_1x_dota15.py) [model](https://drive.google.com/file/d/1AjG3-Db_hmZF1YSKRVnq8j_yuxzualRo/view?usp=sharing) [log](https://drive.google.com/file/d/17dsP9EUbLTV9THkOAA3G3jpmIHHnj83-/view?usp=sharing)        |
|ReDet                      |DOTA-v1.5       |    ReR101-ReFPN     |   -    |   -    |   1x     |  67.62 |    [cfg](configs/ReDet/ReDet_re101_refpn_1x_dota15.py) [model](https://drive.google.com/file/d/1vN4ShOqegn4__QY_hgykota20Qa1mnBQ/view?usp=sharing) [log](https://drive.google.com/file/d/1eKiXI91VudU7rGufdEt526cO8kEm9dAc/view?usp=sharing)        |
|ReDet                      |DOTA-v1.5       |    ReR50-ReFPN     |   ✓    |   ✓    |   1x     |  76.80 |    [cfg](configs/ReDet/ReDet_re50_refpn_1x_dota15_ms.py) [model](https://drive.google.com/file/d/1I1IDmt3juw1sm-CT-zaosVVDldAHYBIO/view?usp=sharing) [log](https://drive.google.com/file/d/1T2Eou26T0mpmP93X_XrFk-AhSicLrgGp/view?usp=sharing)        |
|ReDet                      |HRSC2016        |    ReR50-ReFPN     |   -    |   -    |   3x     |  90.46 |    [cfg](configs/ReDet/ReDet_re50_refpn_3x_hrsc2016.py) [model](https://drive.google.com/file/d/1vTU6OeFD6CX4zkQn7szlgL7Qc_MOZpgC/view?usp=sharing) [log](https://drive.google.com/file/d/1csbm3jop9MGOQt8JaEeBg6TEXOZXY-yo/view?usp=sharing)        |

**Note:**
1. All our models are trained on 4GPUs with a learning rate 0.01. If you train your model with more/fewer GPUs, remember to change the learning rate, e.g., 0.02lr=0.0025lr\*8GPU, 0.0025lr=0.0025lr\*1GPU.
2. If you cannot get access to Google Drive, BaiduYun download link can be found [here](https://pan.baidu.com/s/1RowD1GchTQNfuEGvMmH6bQ) with extracting code **ABCD**.


## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Getting Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage.


## Citation

```BibTeX
@InProceedings{han2021ReDet,
    author    = {Han, Jiaming and Ding, Jian and Xue, Nan and Xia, Gui-Song},
    title     = {ReDet: A Rotation-equivariant Detector for Aerial Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2786-2795}
}
```
