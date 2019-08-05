Training MaskRCNN using Horovod

#### Step 1:

Download the MS-COCO dataset by running the following script

```bash
python mscoco.py
```



#### Step 2:

Run the Mask-RCNN horovod script using following command:

```bash
horovodrun -np 8 -H localhost:8 python -u train_mask_rcnn_horovod.py --num-workers 4 --horovod --amp --lr-decay-epoch 8,10 --epochs 12 --log-interval 1000 --val-interval 12
```

