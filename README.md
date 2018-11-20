# Introduction

Author: Chih-Chung Hsu (cchsu@mail.npust.edu.tw)
Department of Management Information Systems
National Pingtung University of Science and Technology

This code is the implementation of our recent paper released at Setpmeber 2018 -- Learning to Detect Fake Face Images in the Wild (ArXiv: https://arxiv.org/abs/1809.08754)
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
5. Open ResNet_DeepUD.ipynb on the browser and run it. 
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



## Citation
@article{hsu2018learning,
  title={Learning to Detect Fake Face Images in the Wild},
  author={Hsu, Chih-Chung and Lee, Chia-Yen and Zhuang, Yi-Xiu},
  journal={arXiv preprint arXiv:1809.08754},
  month={9},
  year={2018}
}

Hsu, Chih-Chung, Chia-Yen Lee, and Yi-Xiu Zhuang. "Learning to Detect Fake Face Images in the Wild." arXiv preprint arXiv:1809.08754, 24 Sep 2018.
Hsu, Chih-Chung, Chia-Yen Lee, and Yi-Xiu Zhuang. "Learning to Detect Fake Face Images in the Wild." IEEE Intertional Symposium on Computer, Consumer and Control (IS3C), Taichung, Dec. 2018.
