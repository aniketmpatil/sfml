#!/bin/bash

cd /home/aniket/WPI/DR_3DOD/codes/sfml
conda init bash
conda activate sfml
python3 data/prepare_train.py /home/aniket/WPI/DR_3DOD/kitti_raw --dataset-format 'kitti_raw' --dump-root /home/aniket/WPI/DR_3DOD/kitti_dump --width 416 --height 128 --num-threads 4 --static-frame /home/aniket/WPI/DR_3DOD/codes/sfml/data/static_frames.txt