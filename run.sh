#!/bin/bash

python main.py --config configs/largescale/subnetonly/resnet50-usc-unsigned-cub.yaml \
               --multigpu 0 \
               --name transfer-weight-freeze_score-200 \
               --data data \
               --prune-rate 0.5 \
               --transfer "ptr-models/resnet50-usc-unsigned.pth"

               
