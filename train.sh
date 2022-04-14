#!/usr/bin/env bashclear

# office31
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office31 -d Office31 -s A -t W -a resnet50  --log logs/RCE/Office31/Office31_A2W --early 15
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office31 -d Office31 -s A -t D -a resnet50  --log logs/RCE/Office31/Office31_A2D --early 15
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office31 -d Office31 -s D -t A -a resnet50  --log logs/RCE/Office31/Office31_D2A --early 15
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office31 -d Office31 -s W -t A -a resnet50  --log logs/RCE/Office31/Office31_W2A --early 15
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office31 -d Office31 -s D -t W -a resnet50  --log logs/RCE/Office31/Office31_D2W --early 15
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office31 -d Office31 -s W -t D -a resnet50  --log logs/RCE/Office31/Office31_W2D --early 15

##### VisDA2017
CUDA_VISIBLE_DEVICES=1 python train_RCE.py visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --per-class-eval --log logs/RCE/VisDA2017 --early 20 --mu 1

#### Office-Home
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Ar -t Cl -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Ar2Cl --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Ar -t Pr -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Ar2Pr --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Ar -t Rw -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Ar2Rw --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Cl -t Ar -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Cl2Ar --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Cl -t Pr -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Cl2Pr --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Cl -t Rw -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Cl2Rw --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Pr -t Ar -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Pr2Ar --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Pr -t Cl -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Pr2Cl --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Pr -t Rw -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Pr2Rw --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Rw -t Ar -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Rw2Ar --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Rw -t Cl -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Rw2Cl --early 20
CUDA_VISIBLE_DEVICES=1 python train_RCE.py office-home -d OfficeHome -s Rw -t Pr -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Rw2Pr --early 20

