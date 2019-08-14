
**Scripts**: https://github.com/dmlc/gluon-cv/blob/master/scripts/detection/ssd/train_ssd.py

**Model**: 
SSD model with resnet50 as base feature extractor.

**Dataset:** Coco

### Single node: 

**Machine Setup**: 
 - 1 x p3dn.x24large 8 gpus
 - mxnet v1.5, 
 - gluoncv v0.5( nightly build, commit - a0ba841e0401e89516e59c1ee85b38c6f77917ba)

**Command:** `python3 train_ssd.py --gpus 0,1,2,3,4,5,6,7 -j 32 --network resnet50_v1 --data-shape 512 --dataset coco --lr 0.002 --epochs 25`

**Input Image Resolution:** 512 x 512

**batch-size:** 32

**number of worker:** 32

**learning rate:** 0.002

**Time Taken:** 764.568 secs per epoch for 25 epochs

**mean Average Precision(mAP):** 0.216

**Average speed:** 153.13 samples/sec
