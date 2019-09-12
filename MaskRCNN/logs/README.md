- Install MXNet
```bash
pip install mxnet-cu101mkl --upgrade --pre
```
- Install Horovod
```bash
HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir git+https://github.com/horovod/horovod
```

- Install GluonCV
```bash
git clone --recursive https://github.com/dmlc/gluon-cv.git
cd gluon-cv
python setup.py install --with-cython
```

- Open `train_mask_rcnn.py` file in GluonCV repo and add below environment variables:
```bash
vim gluon-cv/scripts/instance/mask_rcnn/train_mask_rcnn.py

# set environ variable
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '28'
```

- Command to run the script:
```bash
// On 1 Node with 8 GPUs:
horovodrun -np 8 -H localhost:8 python -u /home/ubuntu/gluon-cv/scripts/instance/mask_rcnn/train_mask_rcnn.py --num-workers 4 --horovod --amp --lr-decay-epoch 8,10 --epochs 12 --log-interval 10 --val-interval 12 --batch-size 32

// On 4 Node with 8 GPUs each:
horovodrun -np 32 -H localhost:8,x.x.x.x:8,x.x.x.x:8,x.x.x.x:8 python -u /home/ubuntu/gluon-cv/scripts/instance/mask_rcnn/train_mask_rcnn.py --num-workers 4 --horovod --amp --lr-decay-epoch 8,10 --epochs 12 --log-interval 10 --val-interval 12 --batch-size 128
```

Replace the `x` with the IP address of your Node <br>
`--batch-size 32`: If you running on 8 GPUs, then this means 4 images/GPUs (`32/8`)

- Mask R-CNN training with EFA enabled shows almost similar to marginally decrease in throughput. 