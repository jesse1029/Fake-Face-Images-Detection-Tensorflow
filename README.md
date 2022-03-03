# Introduction

Author: Chih-Chung Hsu (cchsu@gs.ncku.edu.tw)
Institute of Data Science
National Chemg Kung University

This code is the implementation of our recent paper released at September 2018 -- Learning to Detect Fake Face Images in the Wild (ArXiv: https://arxiv.org/abs/1809.08754)
and our recent paper published on ICIP 2019 -- Y. Zhuang and C. Hsu, "Detecting Generated Image Based on a Coupled Network with Two-Step Pairwise Learning," 2019 IEEE International Conference on Image Processing (ICIP), Taipei, Taiwan, 2019, pp. 3212-3216.

Any suggestion/problem is welcome. 

- Our GAN synthesizers are based on https://github.com/LynnHo/DCGAN-LSGAN-WGAN-WGAN-GP-Tensorflow
- The final results of the proposed method can be reproduced by executing ResNet_DeepUD.ipynb.
- The results without contrastive loss of the proposed NN architecture can be reproduced by executing ResNet_DeepUD_noContrastive.ipynb.

---

## Fake image generator by GANs

Tensorflow implementation of DCGAN, LSGAN, WGAN and WGAN-GP, and we use DCGAN as the network architecture in all experiments.
DCGAN: [Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/abs/1511.06434)
LSGAN: [Least squares generative adversarial networks](https://pdfs.semanticscholar.org/0bbc/35bdbd643fb520ce349bdd486ef2c490f1fc.pdf)
WGAN: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
WGAN-GP: [Improved Training of Wasserstein GANs](http://arxiv.org/abs/1704.00028)
Forked from https://github.com/LynnHo/DCGAN-LSGAN-WGAN-WGAN-GP-Tensorflow


## Prerequisites
- tensorflow r1.10
- python 3.5


# Usage

## Train and Test
```
1. Generate fake images by different GANs (PGGAN code can be found in https://github.com/tkarras/progressive_growing_of_gans)
2. The generated fake images should be located in "result/celeba_[GANNAME]/*.jpg"
3. Extract aligned CelebA face images from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and put them into "data/img_align_celeba/*.jpg"
4. Start Jupyter notebook (type jupyter notebook on Anaconda console)
5. Open ResNet_DeepUD.ipynb on the browser and run it for IS3C paper results based on contrastive loss (initial work).
   [Recommend!!] Run ResNet_DeepUD_triplet_TwStreaming.ipynb for our ICIP19 paper results based on triplet loss + coupled network.

...
```
## Tensorboard
```
tensorboard --logdir=./logs/sia/
...
```

## Datasets
Celeba should be prepared by yourself in ./data/img_align_celeba/*.jpg
Use train_celeba_dcgan.py to create fake images model using DCGAN
Use test_celeba_dcgan.py to create fake images using DCGAN based on learned model.
They should be putted into result folder like result/celeba_dcgan/*.jpg....

### File List
For ResNet_DeepUD version, you need to prepare the pairwise file list formatted in the following
```
image_path_1 image_path_2 Same_or_not
image_path_3 image_path_4 Same_or_not
image_path_5 image_path_6 Same_or_not
image_path_7 image_path_8 Same_or_not
...
```
where image_path_1 is the path to image 1 and image_path_2 is the path to image 2. The Same_or_not is a label indicator when Same_or_not=1 for images 1 and 2 are the same identity and Same_or_not=0 for another case.

For ResNet_DeepUD_triplet_TwStreaming version, you need to prepare file list formatted in the following:
```
image_path_1 Label
image_path_2 Label
image_path_3 Label
image_path_4 Label

...
```
where Label is the identity ID.

## Citation
@INPROCEEDINGS{8803464, 
author={Y. {Zhuang} and C. {Hsu}}, 
booktitle={2019 IEEE International Conference on Image Processing (ICIP)}, 
title={Detecting Generated Image Based on a Coupled Network with Two-Step Pairwise Learning}, 
year={2019}, 
volume={}, 
number={}, 
pages={3212-3216}, 
keywords={Forgery detection;generative adversarial networks;triplet loss;deep learning;coupled network}, 
doi={10.1109/ICIP.2019.8803464}, 
ISSN={2381-8549}, 
month={Sep.},}

Y. Zhuang and C. Hsu, "Detecting Generated Image Based on a Coupled Network with Two-Step Pairwise Learning," 2019 IEEE International Conference on Image Processing (ICIP), Taipei, Taiwan, 2019, pp. 3212-3216.

Hsu, Chih-Chung, Chia-Yen Lee, and Yi-Xiu Zhuang. "Learning to Detect Fake Face Images in the Wild." IEEE Intertional Symposium on Computer, Consumer and Control (IS3C), Taichung, Dec. 2018.
