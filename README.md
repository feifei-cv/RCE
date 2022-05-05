# RCE
Code release for "Unsupervised domain adaptation via risk-consistent estimators" 

## Prerequisites
- torch>=1.7.0
- torchvision

## Training

VisDA-2017
```
CUDA_VISIBLE_DEVICES=0 python train_RCE.py visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --per-class-eval --log logs/RCE/VisDA2017 --mu=1 --threshold 0.95 --trade-off1 0.8 --early 25
```

Office Home
```
CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Ar -t Cl -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Ar2Cl --early 20
```

Office31
```
CUDA_VISIBLE_DEVICES=0 python train_RCE.py office31 -d Office31 -s A -t W -a resnet50  --log logs/RCE/Office31/Office31_A2W --early 15
```



## Acknowledgement
This code is heavily borrowed from  [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library/), [SSL](https://github.com/YBZh/Bridging_UDA_SSL), and [CST]( https://github.com/Liuhong99/CST). It is our pleasure to acknowledge their contributions.

