#!/bin/zsh
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/inference_2.py --ckpt_dir "/media/gnort/HDD6/Study/face-analysis-lightning/logs/train/runs/2024-01-12_00-19-50/checkpoints" \
                        --csv_path "outputs/answer_2.csv" \
                        --root_dir "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/cropped_faces" \
                        --image_list "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/image_lists/face_test.txt" \
                        --image_dir "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/public_test/cropped_faces" \
                        --attributes "age|gender" \
                        --backbone_name "resnet50" \
                        # --image_path "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/cropped_faces/1.jpg"
                        
