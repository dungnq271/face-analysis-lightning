#!/bin/zsh
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/inference.py --ckpt_path "/media/gnort/HDD6/Study/face-analysis-lightning/logs/train/runs/2024-01-09_09-19-55/checkpoints/epoch_022.ckpt" \
                        --root_dir "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/cropped_faces" \
                        --image_list "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/image_lists/face_test.txt" \
                        # --image_path "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/cropped_faces/1.jpg"
                        
