python train.py  \
--warmUpIter 10000 \
--train-dir /public/home/hpc204712181/food/foodH/train \
--val-dir /public/home/hpc204712181/food/foodH/test \
--nested 100 \
--pretrained --freeze-bn \
--out-dir ./output/models 