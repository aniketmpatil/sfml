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

```
python3 data/prepare_train_data.py /root/datasets/kitti_raw --dump-root /root/datasets/kitti_dump --with-depth --with-pose --static-frame /root/codes/sfml/data/static_frames.txt
```
- pebble
- path
- imageio
- scikit-image
- tqdm
- pyyaml

## Docker Commands

newgrp docker
docker run hello-world
docker ps -a (shows all running containers)
docker images
docker rm i [imageID]
docker rm [containerID]
docker pull pytorch/pytorch:latest
docker run -it pytorch/pytorch:latest
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
docker run -it --rm --gpus all pytorch/pytorch:latest
docker run -it --rm --gpus all -v /home/pear_group/aniket:/root/ pytorch/pytorch:latest


