#!/bin/zsh
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=face_heads_1.yaml
