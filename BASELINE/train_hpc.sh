CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --world_size=2 --folder /public/home/hpc204712181/food/foodH/

