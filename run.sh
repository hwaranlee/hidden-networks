#!/bin/bash

python main.py --config configs/largescale/subnetonly/resnet50-usc-unsigned-cub.yaml \
               --multigpu 0 \
               --name scratch-200 \
               --data data \
               --prune-rate 0.5 \
#               --transfer "ptr-models/resnet50-usc-unsigned.pth"

               
