# sfml
Implementation of SFMLearner Pytorch code

## Prepare Train Data Command:


```
python3 data/prepare_train_data.py /home/aniket/WPI/DR_3DOD/kitti_raw --dump-root /home/aniket/WPI/DR_3DOD/kitti_dump --with-depth --with-pose --static-frame /home/aniket/WPI/DR_3DOD/codes/sfml/data/static_frames.txt
```

## Train Command
```
python3 train.py /home/aniket/WPI/DR_3DOD/kitti_dump -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 3
```