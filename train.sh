#!/usr/bin/env bashclear

# office31
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office31 -d Office31 -s A -t D -a resnet50  --log logs/RCE/Office31/Office31_A2D --early 10
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office31 -d Office31 -s A -t W -a resnet50  --log logs/RCE/Office31/Office31_A2W --early 10
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office31 -d Office31 -s D -t A -a resnet50  --log logs/RCE/Office31/Office31_D2A --early 10
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office31 -d Office31 -s W -t A -a resnet50  --log logs/RCE/Office31/Office31_W2A --early 10
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office31 -d Office31 -s D -t W -a resnet50  --log logs/RCE/Office31/Office31_D2W --early 3
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office31 -d Office31 -s W -t D -a resnet50  --log logs/RCE/Office31/Office31_W2D --early 3

##### Office-Home
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Ar -t Cl -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Ar2Cl --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Ar -t Pr -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Ar2Pr --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Ar -t Rw -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Ar2Rw --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Cl -t Ar -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Cl2Ar --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Cl -t Pr -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Cl2Pr --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Cl -t Rw -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Cl2Rw --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Pr -t Ar -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Pr2Ar --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Pr -t Cl -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Pr2Cl --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Pr -t Rw -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Pr2Rw --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Rw -t Ar -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Rw2Ar --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Rw -t Cl -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Rw2Cl --early 20
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py office-home -d OfficeHome -s Rw -t Pr -a resnet50  --log logs/RCE/OfficeHome/OfficeHome_Rw2Pr --early 20

##### VisDA2017
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --per-class-eval --log logs/RCE/VisDA2017 --early 20 --mu=1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --per-class-eval --log logs/RCE/VisDA2017 --early 20 --mu 1 --lr 0.005
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --per-class-eval --log logs/RCE/VisDA2017 --early 20 --mu=1 --temperature 1.8

# Domainnet mu=3,2,1
CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s c -t i -a resnet50 --early 20  --log logs/RCE/DomainNet/DomainNet_c2i --mu 3 --trade-off3 0.1
CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s i -t p -a resnet50 --early 20  --log logs/RCE/DomainNet/DomainNet_i2p --mu 3 --trade-off3 0.1
CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s p -t q -a resnet50 --early 20  --log logs/RCE/DomainNet/DomainNet_p2q --mu 3 --trade-off3 0.1
CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s q -t r -a resnet50 --early 20  --log logs/RCE/DomainNet/DomainNet_q2r --mu 3 --trade-off3 0.1
CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s r -t s -a resnet50 --early 20  --log logs/RCE/DomainNet/DomainNet_r2s --mu 3 --trade-off3 0.1
CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s s -t c -a resnet50 --early 20  --log logs/RCE/DomainNet/DomainNet_s2c --mu 3 --trade-off3 0.1


###########
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s c -t i -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_c2i --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s c -t p -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_c2p --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s c -t q -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_c2q --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s c -t r -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_c2r --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s c -t s -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_c2s --mu 1
#
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s i -t c -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_i2c --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s i -t p -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_i2p --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s i -t q -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_i2q --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s i -t r -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_i2r --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s i -t s -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_i2s --mu 1
#
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s p -t c -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_p2c --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s p -t i -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_p2i --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s p -t q -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_p2q --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s p -t r -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_p2r --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s p -t s -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_p2s --mu 1
#
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s q -t c -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_q2c --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s q -t i -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_q2i --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s q -t p -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_q2p --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s q -t r -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_q2r --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s q -t s -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_q2s --mu 1
#
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s s -t c -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_s2c --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s s -t i -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_s2i --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s s -t p -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_s2p --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s s -t q -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_s2q --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s s -t r -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_s2r --mu 1
#
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s r -t c -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_r2c --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s r -t i -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_r2i --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s r -t p -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_r2p --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s r -t q -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_r2q --mu 1
#CUDA_VISIBLE_DEVICES=0 python train_RCE.py domainnet -d DomainNet -s r -t s -a resnet101 --early 20  -i 2500 -p 500  --bottleneck-dim 1024  --log logs/RCE/DomainNet/DomainNet_r2s --mu 1
