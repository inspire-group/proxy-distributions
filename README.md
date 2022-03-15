# Robust Learning Meets Generative Models: Can Proxy Distributions Improve Adversarial Robustness? 

Pdf: [Arxiv](https://arxiv.org/abs/2104.09425), [Openreview](https://openreview.net/forum?id=WVX0NNVBBkV&noteId=05ntgCksbhL)


Code for our **ICLR 2022** paper where we show that synthetic data from diffusion models can provide a tremendous boost in the performance of robust training. We also provide synthetic data used in the paper for all five datasets, namely CIFAR-10, CIFAR-100, ImageNet, CelebA, and AFHQ. We also provide synthetic data from *seven* different generative models for CIFAR-10, which was used to analyze impact of different generative models in section 3.2. 

Despite being minimalistic, this codebase also offers *multi-node and multi-gpu* adversarial training support.  


## Getting started

Let's start by installing all dependencies. 

* `pip install torch torchvision easydict`
* `pip install git+https://github.com/RobustBench/robustbench`
* `pip install git+https://github.com/fra31/auto-attack`



## Training a robust classifier

We can perform adversarial training on four GPUs using the following command.

`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=6116 train.py --dataset cifar10 --syn-data-list ddpm_cifar10 --syn-data-dir dir_where_syn_data_is_stored --arch wrn_34_10 --trainer pgd --val-method pgd --batch-size 128 --batch-size-syn 128 --exp-name name_of_experiment`

* `--trainer`: Choice of training method from (`baseline, trades, pgd, fgsm`). `baseline` refer to benign training, i.e., no adversary is present, while other three are variations of adversarial training. 
* `--val-method`: Choice of training method from (`baseline, pgd, auto`). In evaluation, we can also use either `baseline` or an adversarial attack from `pgd` or `autoattack`. Note that we use a subset of attacks in `autoattack` for faster evaluation. 
* `--syn-data-list`: Choice of synthetic dataset. It will be loaded from the directory provided by `--syn-data-dir` flag. One can choose to provide multiple synthetic datasets.  
* `--dataset`: Choice of dataset from `(cifar-10, cifar-100)`

We prvoide the synthetic dataloader and its integration with real data in `utils.py`. While the current demo consider cifar-10 and cifar-100 as the primary use cases, it can be easily extended to other datasets. 


Now let's consider the variations of aforementioned setup. 

1. *Training on a single-gpu*: Just specify the `--no-dist` flag. 

`CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --syn-data-list ddpm_cifar10 --syn-data-dir dir_where_syn_data_is_stored --arch wrn_34_10 --trainer pgd --val-method pgd --batch-size 128 --batch-size-syn 128 --exp-name name_of_experiment --no-dist`

2. *Training without synthetic data*: Simply drop the synthetic data args. 

`CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --arch wrn_34_10 --trainer pgd --val-method pgd --batch-size 128 --exp-name name_of_experiment --no-dist`

3. *Multi-node training*: We will use slurm to lauch processes on different nodes. We provide an example script (`./scripts/train.slurm`) to adversarially train a ResNest-152 model on eight nodes with two gpus per node. Note that beyond the cmd line args with multi-node are identical to single-node, thus the only requirement for multi-node is to launch it using slurm. 

## Pre-trained models
Our pre-trained models are easily accessible through [RobustBench](https://robustbench.github.io/). For example, one can load our robust `WideResNet-34-10` model for CIFAR-10 (Linf) dataset using following two lines.

```python
from robustbench.utils import load_model
model = load_model(model_name='Sehwag2021Proxy', dataset='cifar10', threat_model='Linf')
```

## Synthetic data zoo

We release synthetic images for all five datasets used in our paper. You can download them using the link provided below. Class-conditioning implies that the generative model itself generates labelled images. With generative models that are not class-conditioned, we use an additional classifier to generate pseudo labels, as described in the paper.  


|  Dataset  	| Number of training images in dataset 	| Generative model 	| Number of synthetic images  	| Class-conditioned 	| Samples	|
|:---------:	|:-------------------------------------:	|:----------------:	|:----------------------------:	|:-----------------:	|:-----------------:	|
|  CIFAR-10 	|                  50K                  	|    [DDPM](https://arxiv.org/abs/2006.11239)   	|              10M             	|       No       	|       [Link](https://drive.google.com/drive/folders/1xEJFc3OfXnClkm5-zLExiov1jUsnlaRK?usp=sharing)       	|
|        "   	|                "                       	|     [StyleGAN](https://github.com/NVlabs/stylegan2-ada)     	|              10M             	|       Yes       	|       [Link](https://drive.google.com/file/d/1HvOBP7mmDImGudzjTinMzB-R3dVFGLr9/view?usp=sharing)       	|
|        "   	|               "                        	|     [WGAN-ALP](https://arxiv.org/pdf/1907.05681v3.pdf)     	|              1M              	|       No       	|       [Link](https://drive.google.com/file/d/183khLDL1xMdNetsHxEmICNRhJrsbsbVV/view?usp=sharing)       	|
|          " 	|               "                       	|       [E2GAN](https://arxiv.org/abs/2007.09180)      	|              1M              	|       No       	|       [Link](https://drive.google.com/file/d/1v9KBwmmmyz0SW378iZKCIdWik3Q5-lUj/view?usp=sharing)       	|
|          " 	|              "                         	|   [DiffCrBigGAN](https://arxiv.org/abs/2006.10738)   	|              1M              	|       Yes       	|       [Link](https://drive.google.com/file/d/1wxMkuSfC4IdX4Ay2LUSWC0TK1Vcyv6Nk/view?usp=sharing)       	|
|         "  	|                "                       	|        [NDA](https://arxiv.org/abs/2102.05113)       	|              1M              	|       Yes       	|       [Link](https://drive.google.com/file/d/1Iom7SBTZhF6NyHsy-VrkIgUcBynXp3PJ/view?usp=sharing)       	|
|         "  	|              "                         	|    [DiffBigGAN](https://arxiv.org/abs/2006.10738)    	|              1M              	|       Yes       	|       [Link](https://drive.google.com/file/d/1TfLhVYqQW8HqK5t9CF70Yghe2HJ2mvbh/view?usp=sharing)       	|
|   CelebA  	|           120K                            	|    [StyleFormer](https://arxiv.org/abs/2106.07023v2)   	|              1M              	|       No       	|       [Link](https://drive.google.com/file/d/1qUp0ZradTDCfxuJkwRuWLk1LhyKQXvmb/view?usp=sharing)       	|
|      "     	|             "                          	|       [DDPM](https://arxiv.org/abs/2102.09672)       	|              1M              	|       No       	|       [Link](https://drive.google.com/file/d/1st02bZMziKcWl05XVyvS9TTIaNtuDxdb/view?usp=sharing)       	|
|  ImageNet 	|                  1.2M                 	|      [BigGAN](https://arxiv.org/abs/1809.11096v2)      	|              1M              	|       Yes       	|       [Link](https://drive.google.com/file/d/1gwFgkDRRfWgn6ylWXfYDZnTbaz2Ur3wz/view?usp=sharing)       	|
|       "    	|              "                         	|       [DDPM](https://arxiv.org/abs/2102.09672)       	|             400K             	|       Yes       	|       [Link](https://drive.google.com/file/d/1_IuH36YmeHiNc0WMXqw0ZvqKaOiX5I2G/view?usp=sharing)       	|
| CIFAR-100 	|                  50K                  	|       [DDPM](https://arxiv.org/abs/2006.11239)       	|              1M              	|       Yes       	|       [Link](https://drive.google.com/file/d/1k20VkxXCxIR7dKPjud3YrmH7svQuuUZo/view?usp=sharing)       	|
|    AFHQ   	|            15K                           	|     [StyleGAN](https://arxiv.org/abs/2006.06676)     	|           300K                   	|       Yes       	|       [Link](https://drive.google.com/file/d/15-q79b4Gga6dQbjvh3xSQYx8WA3X3cZM/view?usp=sharing)       	|



## Reference
If you find this work helpful, consider citing it. 

```bibtex
@inproceedings{sehwag2022robust,
    title={Robust Learning Meets Generative Models: Can Proxy Distributions Improve Adversarial Robustness?},
    author={Vikash Sehwag and Saeed Mahloujifar and Tinashe Handina and Sihui Dai and Chong Xiang and Mung Chiang and Prateek Mittal},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=WVX0NNVBBkV}
}
}
```