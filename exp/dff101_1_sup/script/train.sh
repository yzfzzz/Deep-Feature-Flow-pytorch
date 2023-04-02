#!/bin/bash
# ”yezifeng/segmentation/structure-guided“ is your path
export PYTHONPATH=$PYTHONPATH:/home/yezifeng/anaconda3/envs/video/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/home/yezifeng/segmentation/DFF
export PYTHONPATH=$PYTHONPATH:/home/yezifeng/segmentation/DFF/lib/model
export PYTHONPATH=$PYTHONPATH:/home/yezifeng/segmentation/structure-guided/IFR/mmseg

cd /home/yezifeng/segmentation/DFF && \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM exp/dff101_1_sup/python/train.py \
        --exp_name dff101_1_sup \
        --weight_res101 ./pretrained/psp_101.pth \
        --weight_flownet ./pretrained/flownet_pretrained.pth \
        --lr 0.0005 \
        --train_batch_size 2 \
        --train_num_workers 2 \
        --test_batch_size 1 \
        --test_num_workers 2 \
        --train_iterations 40000 \
        --log_interval 50 \
        --val_interval 1000 \
        --work_dirs /home/yezifeng/segmentation/structure-guided/Accel/work_dirs \