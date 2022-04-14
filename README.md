# RCE
Code release for "Unsupervised domain adaptation via risk-consistent estimators" 

## Prerequisites
- torch>=1.7.0
- torchvision

## Training

VisDA-2017
```
CUDA_VISIBLE_DEVICES=0 python train_RCE.py visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --per-class-eval --log logs/RCE/VisDA2017 --early 20 --mu 1
```

Office Home
```
CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Ar -t Cl -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Ar2Cl --early 20
```


## Acknowledgement
This code is implemented based on the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library/), and it is our pleasure to acknowledge their contributions.

The SAM code is adapted from [https://github.com/davda54/sam](https://github.com/davda54/sam).

