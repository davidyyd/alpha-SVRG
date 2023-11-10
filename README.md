#  &alpha;-SVRG
Official PyTorch implementation for $\alpha$-SVRG

> [**A Coefficient Makes SVRG Effective**](https://arxiv.org/pdf/2311.05589.pdf)<br>
> [Yida Yin](https://davidyyd.github.io), [Zhiqiu Xu](https://oscarxzq.github.io), [Zhiyuan Li](https://zhiyuanli.ttic.edu/),[Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Zhuang Liu](https://liuzhuang13.github.io)
> <br>UC Berkeley, University of Pennsylvania, Toyota Technological Institute at Chicago, and Meta AI Research<br>
>[[Paper]](https://arxiv.org/abs/2311.05589) [[Video]](https://iclr.cc/virtual/2025/poster/28009) [[Project page]](https://davidyyd.github.io/alpha-SVRG/)
<p align="center">
<img src="https://github.com/davidyyd/alpha-SVRG/assets/91447088/c88e671c-ec7c-4b79-a6bd-b3c6f7e5908c"
class="center">
</p>


We introduce $\alpha$-SVRG: applying a linearly decaying coefficient $\alpha$ to control the strength of the variance reduction term in SVRG.

## Results 

### Smaller Datasets

**Train Loss**
| | CIFAR-100 | Pets | Flowers | STL-10 | Food-101 | DTD | SVHN | EuroSAT |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline      | 2.66 | 2.20 | 2.40 | 1.64 | 2.45 | 1.98 | 1.59 | 1.25 | 
| SVRG          | 2.94 | 3.42 | 2.26 | 1.90 | 3.03 | 2.01 | 1.64 | 1.25 |
| $\alpha$-SVRG | **2.62** | **1.96** | **2.16** | **1.57** | **2.42** | **1.83** | **1.57** | **1.23** |

**Validation Accuracy** 
| | CIFAR-100 | Pets | Flowers | STL-10 | Food-101 | DTD | SVHN | EuroSAT |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline      | 81.0 | 72.8 | 80.8 | 82.3 | **85.9** | 57.9 | 94.9 | 98.1 |
| SVRG          | 78.2 | 17.6 | 82.6 | 65.1 | 79.6 | 57.8 | 95.7 | 97.9 |
| $\alpha$-SVRG | **81.4** | **77.8** | **83.3** | **84.0** | **85.9** | **61.8** | **95.8** | **98.2** |

$\alpha$-SVRG improves both the train loss and the validation accuracy across all small datasets, but the standard SVRG mostly hurts the performance.
### ImageNet-1K

**Train Loss**
| | ConvNeXt-F | ViT-T | Swin-F | Mixer-S | ViT-B | ConvNeXt-B |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline      | 3.487 | 3.443 | 3.427 | 3.635 | 2.817 | 2.644 |
| SVRG          | 3.505 | 3.431 | **3.389** | 3.776 | 2.309 | 3.113 |
| $\alpha$-SVRG | **3.467** | **3.415** | 3.392 | **3.609** | **2.806** | **2.642** |

**Validation Accuracy** 
| | ConvNeXt-F | ViT-T | Swin-F | Mixer-S | ViT-B | ConvNeXt-B |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline      | 76.0 | 73.9 | 74.3 | **71.0** | **81.6** | **83.7** |
| SVRG          | 75.7 | **74.3** | 74.3 | 68.8 | 78.0 | 80.8 |
| $\alpha$-SVRG | **76.3** | 74.2 | **74.8** | 70.5 | **81.6** | 83.1 |

$\alpha$-SVRG consistently decreases the train loss, whereas standard SVRG increases it in most models. Note that a lower training loss in α-SVRG does not always lead to better generalization on the validation set. This is out of scope for $\alpha$-SVRG as an optimization method, but warrant future research on co-adapting optimization and regularization.

## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Training

We list commands for $\alpha$-SVRG on `convnext_femto` and `vit_base` with coefficient `0.75`.
- For training other models, change `--model` accordingly, e.g., to `vit_tiny`, `convnext_base`, `vit_base`.
- For using different coefficients, change `--coefficient` accordingly, e.g., to `1`, `0.5`.
- `--use_cache_svrg` can be enabled on smaller models provided with sufficient memory and disabled on larger models.
- Our results of smaller models on ImageNet-1K were produced with 4 nodes, each with 8 gpus. Our results of smaller models on ImageNet-1K were produced with 8 nodes, each with 8 gpus. Our results of ConvNeXt-Femto on small datasets were produced with 8 gpus. 

Below we give example commands for both smaller models and larger models on ImageNet-1K and ConvNeXt-Femto on small datasets.

**Smaller models**

```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_femto --epochs 300 \
--batch_size 128 --lr 4e-3 \
--use_svrg true --coefficient 0.75 --svrg_schedule linear --use_cache_svrg true \
--data_path /path/to/data/ --data_set IMNET \
--output_dir /path/to/results/
```

**Larger models**

```
python run_with_submitit.py --nodes 8 --ngpus 8 \
--model vit_base --epochs 300 \
--batch_size 64 --lr 4e-3 \
--use_svrg true --coefficient 0.75 --svrg_schedule linear \
--data_path /path/to/data/ --data_set IMNET \
--output_dir /path/to/results/
```

**ConvNeXt-Femto on small datasets**
- Fill in `epochs`, `warmup_epochs`, and `batch_size` based on `data_set`.
- Note that `batch_size` is the batch size for each gpu.
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_femto --epochs $epochs --warmup_epochs $warmup_epochs \
--batch_size $batch_size --lr 4e-3 \
--use_svrg true --coefficient 0.75 --svrg_schedule linear --use_cache_svrg true \
--data_path /path/to/data/ --data_set $data_set \
--output_dir /path/to/results/
```
## Evaluation

single-GPU
```
python main.py --model convnext_femto --eval true \
--resume /path/to/model \
--data_path /path/to/data
```

multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_femto --eval true \
--resume /path/to/model \
--data_path /path/to/data
```


## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) codebase.

## Citation
If you find this repository helpful, please consider citing:
```bibtex
@inproceedings{yin2023coefficient,
      title={A Coefficient Makes SVRG Effective}, 
      author={Yida Yin and Zhiqiu Xu and Zhiyuan Li and Trevor Darrell and Zhuang Liu},
      year={2025},
      booktitle={ICLR},
}
```
