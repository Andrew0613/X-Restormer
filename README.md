# X-Restormer [[Paper Link]](https://arxiv.org/abs/2310.11881) 

### A Comparative Study of Image Restoration Networks for General Backbone Network Design
[Xiangyu Chen*](https://chxy95.github.io/), [Zheyuan Li*](https://xiaom233.github.io/), [Yuandong Pu*](https://andrew0613.github.io/),[Yihao Liu](https://scholar.google.com/citations?user=WRIYcNwAAAAJ&hl=zh-CN&oi=ao), [Jiantao Zhou](https://www.fst.um.edu.mo/personal/jtzhou/), [Yu Qiao](https://mmlab.siat.ac.cn/yuqiao) and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)

#### BibTeX

    @article{chen2023comparative,
      title={A Comparative Study of Image Restoration Networks for General Backbone Network Design}, 
      author={Chen, Xiangyu and Li, Zheyuan and Pu, Yuandong and Liu, Yihao and Zhou, Jiantao and Qiao, Yu and Dong, Chao},
      journal={arXiv preprint arXiv:2310.11881},
      year={2023}
    }

<!--## Updates
- 2023-10-18: Release the first version of the paper at Arxiv.
- 2023-10-19: Release the codes, models and results.-->


<!-- ## Overview
<img src="figures/Structure.png" width="600"/>
<img src="figures/relative_performance.jpg" width="600"/> -->

<!-- ## Visual Comparison
**Visual Comparison on SR.**

<img src="figures/visual_sr.png" width="800"/>

**Visual Comparison on Denoising.**

<img src="figures/visual_denoise.png" width="800"/>

**Visual Comparison on Deblurring.**

<img src="figures/visual_deblur.png" width="800"/>

**Visual Comparison on Deraining.**

<img src="figures/visual_derain.png" width="800"/>

**Visual Comparison on Dehazing.**

<img src="figures/visual_dehaze.png" width="800"/> -->

## Environment
- [PyTorch>=1.13.0](https://pytorch.org/) **(Recommend **NOT** using torch 1.8!)**
- [BasicSR==1.4.2](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md)
### Installation
Install Pytorch first.
Then,
```
pip install -r requirements.txt
python setup.py develop
```

## How To Test

- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.
- The pretrained models are available at
[Google Drive](https://drive.google.com/drive/folders/16WxegSAN_sescgrfW4ZMO4b6TcR_7T24?usp=share_link) or [Baidu Netdisk](https://pan.baidu.com/s/1OvyRe6u08HXFQI8NACOhdg?pwd=im3q) (access code: im3q).
- Then run the following codes (taking `sr_300k.pth` as an example):
```
python xrestormer/test.py -opt options/test/001_xrestormer_sr.yml
```
The testing results will be saved in the `./results` folder.

- Refer to `./options/test/001_xrestormer_sr.yml` for **inference** without the ground truth image.


## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1231 xrestormer/train.py -opt ./options/train/001_xrestormer_sr.yml --launcher pytorch
```
- Note that the default batch size per GPU is 4, which will cost about 60G memory for each GPU.

The training logs and weights will be saved in the `./experiments` folder.

## Results
The inference results on benchmark datasets are available at
[Google Drive](https://drive.google.com/drive/folders/17gzfSKySkQd4iUjMZfk5zuEfm12T3U6Y?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1LaTGD-x66-QvZ9WE0QhFxA?pwd=g9dw) (access code: g9dw).


## Contact
If you have any question, please contact puyuandong01061313@gmail.com or chxy95@gmail.com.
