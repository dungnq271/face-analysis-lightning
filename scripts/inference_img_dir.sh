#!/bin/zsh
python src/inference.py \
       --ckpt-path checkpoints/face_256_e99.ckpt \
       --pth-path face_256_e99.pth \
       --image-dir data/final/aligned_faces_test_20/images \
       --batch-size 128 \
       --name2id-json public_test/public_test_and_submission_guidelines/file_name_to_image_id.json \
       --existing-csv public_test/Answers/answer.csv
       

