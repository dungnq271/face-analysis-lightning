#!/bin/zsh
python src/inference.py \
       --ckpt-path checkpoints/face_256_e99.ckpt \
       --pth-path face_256_e99.pth \
       --image-path data/final/aligned_faces_test_20/images/1.jpg \
       --batch-size 32       
       

