#!/usr/bin/bash
echo "<<3D Shape Correspondence Test Start>>"

python3.8 experiments/functional_correspondence/functional_correspondence.py --test_dataset=faust --input_feature=hks --load_model=experiments/functional_correspondence/pretrained_models/starlab_hks.pth --evaluate
python3.8 experiments/functional_correspondence/find_model_correspondence.py --evaluate --test_dataset=psb --input_features=hks --load_model=experiments/functional_correspondence/pretrained_models/starlab_hks.pth --num_test=10 --top_k=3
