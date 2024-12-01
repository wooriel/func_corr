from test3_dataset import PSBTestDataset
from diffusion_net.utils import toNP
import diffusion_net
import os
import sys
import argparse
import random
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import torch
import torch.nn
from torch.utils.data import DataLoader

from fmaps_model import FunctionalMapCorrespondenceWithDiffusionNetFeaturesWithoutVts

# add the path to the DiffusionNet src
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))


# === Options


# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true",
                    help="evaluate using the pretrained model")
parser.add_argument("--test_dataset", type=str,
                    default="faust", help="what dataset to test on")
parser.add_argument("--input_features", type=str,
                    help="what features to use as input ('xyz' or 'hks') default: hks", default='hks')
parser.add_argument("--load_model", type=str,
                    help="path to load a pretrained model from")
# parser.add_argument("--num_test", type=int, default=10,
#                     help="number of test to find similar mesh")
parser.add_argument("--top_k", type=int, default=3,
                    help="number of similar model to retrieve")
args = parser.parse_args()

# system things
device = torch.device('cuda')
dtype = torch.float32

# model
input_features = args.input_features  # one of ['xyz', 'hks']
k_eig = 128

# functional maps settings
n_fmap = 30  # number of eigenvectors used within functional maps
n_feat = 128  # dimension of features computed by DiffusionNet extractor
lambda_param = 1e-3  # functional map block regularization parameter

# training settings
train = not args.evaluate
n_epoch = 50 # 5
lr = 5e-4
decay_every = 9999
decay_rate = 0.1
augment_random_rotate = (input_features == 'xyz')


# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", args.test_dataset, "op_cache")
geodesic_cache_dir = os.path.join(
    base_path, "data", args.test_dataset, "geodesic_cache")  # for evaluating error metrics
model_save_path = os.path.join(
    base_path, "saved_models/{}_{}.pth".format(args.test_dataset, input_features))
dataset_path = os.path.join(base_path, "data")
diffusion_net.utils.ensure_dir_exists(os.path.join(base_path, "saved_models/"))
# Test paths
test_path = os.path.join(base_path, "test", args.test_dataset)
t_sim_path = os.path.join(test_path, "sim_model")
diffusion_net.utils.ensure_dir_exists(os.path.join(base_path, "test/"))
diffusion_net.utils.ensure_dir_exists(t_sim_path)


# === Load datasets
test_dataset = PSBTestDataset(dataset_path, name=args.test_dataset,
                                k_eig=k_eig, n_fmap=n_fmap, use_cache=True, op_cache_dir=op_cache_dir)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=None, shuffle=False)


# === Create the model

C_in = {'xyz': 3, 'hks': 16}[input_features]  # dimension of input features

model = FunctionalMapCorrespondenceWithDiffusionNetFeaturesWithoutVts(
    n_feat=n_feat,
    n_fmap=n_fmap,
    input_features=input_features,
    lambda_param=lambda_param
)


model = model.to(device)

if args.load_model:
    # load the pretrained model
    print("Loading pretrained model from: " + str(args.load_model))
    model.load_state_dict(torch.load(args.load_model))
    print("...done")

if args.evaluate and not args.load_model:
    raise ValueError(
        "Called with --evaluate but not --load_model. This will evaluate on a randomly initialized model, which is probably not what you want to do.")


# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Do an evaluation pass on the test dataset
def test(with_geodesic_error=False):

    model.eval()

    t_src_idx = test_dataset.get_sidx()
    name_list = test_dataset.get_names()
    n1_idx = defaultdict(lambda: [-1, 0])
    max_diag = defaultdict(list) # stores C_pred diagonal sum and name2

    # Make torch that saves absolute diagonal sum of source model
    num_class = 13 # 13
    sim_score = torch.zeros((num_class, 200-num_class))
    

    with torch.no_grad():

        cidx = 0
        for data in tqdm(test_loader):

            # Get data
            shape1, shape2 = data # , C_gt
            *shape1, name1 = shape1
            *shape2, name2 = shape2
            shape1, shape2 = [x.to(device) for x in shape1], [
                x.to(device) for x in shape2]

            verts1_orig = shape1[0]
            if augment_random_rotate:
                shape1[0] = diffusion_net.utils.random_rotate_points(shape1[0])
                shape2[0] = diffusion_net.utils.random_rotate_points(shape2[0])

            # Apply the model
            C_pred, feat1, feat2 = model(shape1, shape2)
            C_pred = C_pred.squeeze(0)

            rprob_corr, cprob_corr = diffusion_net.geometry.get_prob_fmap(C_pred)

            # Calculate Diagonal
            C_diag_r = rprob_corr.diagonal(0)
            dsum_r = C_diag_r.abs().sum()
            max_diag[name1].append(name2)

            # set index for source mesh
            if n1_idx[name1] == [-1, 0]:
                n1_idx[name1][0] = cidx
                cidx += 1

            sim_score[n1_idx[name1][0], n1_idx[name1][1]] = dsum_r
            n1_idx[name1][1] += 1


    return sim_score, max_diag, n1_idx


# Test
print("<<Test 3 Start>>")
sim_score, max_diag, n1_idx = test(with_geodesic_error=False)

_k = list(max_diag.keys())
count = 0
sim_score.to(device)
for i in range(len(_k)):
    score = 0
    n1 = _k[i]
    sim_idx = n1_idx[n1][0] # n1_idx가 sim_idx에서 n1이 차지하는 row, n1열에 값 넣을때 track하는 column임(200)
    top_k, ind_topk = sim_score[sim_idx].topk(args.top_k)
    top_k = toNP(top_k)
    ind_topk = toNP(ind_topk)
    print("Test {}-{}:".format(i+1, n1), end=' ')
    diag_n2 = max_diag[n1] # list of name2
    for j in range(len(top_k)):
        val = top_k[j]
        # find name
        n2 = diag_n2[ind_topk[j]]
        if j == len(top_k)-1:
            print("{}".format(n2))
        else:
            print("{},".format(n2), end=" ")
        score += diffusion_net.geometry.equal_class(n1, n2)
    if score >= args.top_k/2:
        count += 1
percent = round(count/(len(_k))*100
print("Evaluated {} out of {} tests, {}%".format(count, len(_k), round(count/(len(_k))*100)))
if percent >= 90:
    print("PASS")
else:
    print("FAIL")
print("<<Test 3 Finished>>")