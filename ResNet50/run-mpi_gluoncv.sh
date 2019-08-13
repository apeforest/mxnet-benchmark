mpirun -np 256 --hostfile ~/efs/hosts_hvd \
	-map-by ppr:4:socket -mca pml ob1 -mca btl ^openib  -mca btl_tcp_if_include ens5 -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=ens5 \
	-x NCCL_MIN_NRINGS=8 -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x MXNET_SAFE_ACCUMULATION=0 \
	python /home/ubuntu/efs/resnet50/train_imagenet_hvd.py --model resnet50_v1b \
		--dtype float16 --batch-size 128 --lr 12.8 --lr-mode poly -j 4 \
		--warmup-epochs 5 --num-epochs 91 --last-gamma \
		--use-rec --label-smoothing --no-wd \
		--rec-train /home/ubuntu/mxnet-data/imagenet/train-480px-q95.rec \
		--rec-train-idx /home/ubuntu/mxnet-data/imagenet/train-480px-q95.idx \
		--rec-val /home/ubuntu/mxnet-data/imagenet/val-480px-q95.rec \
		--rec-val-idx /home/ubuntu/mxnet-data/imagenet/val-480px-q95.idx \
		--log-interval 0 \
		--mode hybrid 2>&1
