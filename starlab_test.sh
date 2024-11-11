#!/bin/bash
echo "<<3D Shape Correspondence Test Start>>"

python3.8 experiments/functional_correspondence/functional_correspondence.py --test_dataset=faust --input_feature=hks --load_model=experiments/functional_correspondence/saved_models/faust_hks.pth --evaluate
