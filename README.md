## Official code for *Hessian Aware Low-Rank Perturbation for Order-Robust Continual Learning* (HALRP, accepted by IEEE TKDE)

[[paper](https://ieeexplore.ieee.org/document/10572323)][[arxiv (latest)](https://arxiv.org/abs/2311.15161)][[project page](https://lijiaqi.github.io/projects/tkde-HALRP.html)]


### Experiments

- Prepare datasets:
```shell
bash scripts-prepare/download_tinyimgnet.sh
bash scripts-prepare/download_others.sh
```

- Run HALRP on ```5-dataset``` with AlexNet/ResNet18:
```shell
bash scripts/five_AlexNet_HALRP.sh
bash scripts/five_ResNet18_HALRP.sh
```

- Run HALRP on ```TinyImageNet 20-split``` with AlexNet/Resnet18:
```shell
bash scripts/tiny20_AlexNet_HALRP.sh
bash scripts/tiny20_ResNet18_HALRP.sh
```

- Run HALRP on ```TinyImageNet 40-split``` with AlexNet/Resnet18:
```shell
bash scripts/tiny40_AlexNet_HALRP.sh
bash scripts/tiny40_ResNet18_HALRP.sh
```

- Run HALRP on ```CIFAR100-Splits/-SuperClass``` with LeNet:
```shell
### 'TRAINSIZE=1.0' means "100% of training data. 
###   Change this percentage to reproduce the results in Table 2&5 of the paper.
bash scripts/cifar100_splits100_LeNet_HALRP.sh A # (or B/C/D/E for other task orders)
bash scripts/cifar100_super100_LeNet_HALRP.sh A # (or B/C/D/E for other task orders)
```

- Run HALRP on ```PMNIST``` with LeNet:
```shell
bash scripts/pmnist_LeNet_HALRP.sh
```

### Citation
If you find it useful for your study, please consider to cite:
```
@ARTICLE{li2024hessian,
  author={Li, Jiaqi and Lai, Yuanhao and Wang, Rui and Shui, Changjian and Sahoo, Sabyasachi and Ling, Charles X. and Yang, Shichun and Wang, Boyu and Gagné, Christian and Zhou, Fan},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Hessian Aware Low-Rank Perturbation for Order-Robust Continual Learning}, 
  year={2024},
  volume={36},
  number={11},
  pages={6385-6396},
  doi={10.1109/TKDE.2024.3419449}
}
```
### Acknowledgement
This work was finished with [Dr. Fan Zhou](https://fzhou.cc/)(@Beihang University) and [Prof. Christian Gagné](https://chgagne.github.io/english/)(@Université Laval).