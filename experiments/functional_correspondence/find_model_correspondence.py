# from faust_scape_dataset import FaustScapeDataset
# from psb_dataset import PSBDataset
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
# parser.add_argument("--train_dataset", type=str,
#                     default="faust", help="what dataset to train on")
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
# parser.add_argument("--result_path", type=str, help="path to save a result correspondence")
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
# if args.train_dataset == "psb" or args.test_dataset == "psb":
#     op_cache_dir = os.path.join(base_path, "data", "psb", "op_cache")
#     geodesic_cache_dir = os.path.join(base_path, "data", "psb", "geodesic_cache")
model_save_path = os.path.join(
    base_path, "saved_models/{}_{}.pth".format(args.test_dataset, input_features))
dataset_path = os.path.join(base_path, "data")
diffusion_net.utils.ensure_dir_exists(os.path.join(base_path, "saved_models/"))
# Ground Truth paths
gt_path = os.path.join(dataset_path, "gt_fmap")
# Test paths
test_path = os.path.join(base_path, "test", args.test_dataset)
t_sim_path = os.path.join(test_path, "sim_model")
diffusion_net.utils.ensure_dir_exists(os.path.join(base_path, "test/"))
diffusion_net.utils.ensure_dir_exists(t_sim_path)


# === Load datasets
# test_dataset = PSBTestDataset(dataset_path, name=args.test_dataset, n_trial=args.num_test,
#                                 k_eig=k_eig, n_fmap=n_fmap, use_cache=True, op_cache_dir=op_cache_dir)
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


# def train_epoch(epoch):

#     # Implement lr decay
#     if epoch > 0 and epoch % decay_every == 0:
#         global lr
#         lr *= decay_rate
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

#     # Set model to 'train' mode
#     model.train()
#     optimizer.zero_grad()

#     losses = []

#     for data in tqdm(train_loader):

#         # Get data
#         shape1, shape2, C_gt = data
#         *shape1, name1 = shape1
#         *shape2, name2 = shape2
#         shape1, shape2, C_gt = [x.to(device) for x in shape1], [x.to(
#             device) for x in shape2], C_gt.to(device).unsqueeze(0)

#         # Randomly rotate positions
#         if augment_random_rotate:
#             shape1[0] = diffusion_net.utils.random_rotate_points(shape1[0])
#             shape2[0] = diffusion_net.utils.random_rotate_points(shape2[0])

#         # Apply the model
#         C_pred, feat1, feat2 = model(shape1, shape2)

#         # Evaluate loss
#         loss = torch.mean(torch.square(C_pred-C_gt))  # L2 loss
#         losses.append(toNP(loss))
#         loss.backward()

#         # Step the optimizer
#         optimizer.step()
#         optimizer.zero_grad()

#     train_loss = np.mean(losses)

#     return train_loss


# Do an evaluation pass on the test dataset
def test(with_geodesic_error=False):

    # if with_geodesic_error:
    #     print("Evaluating geodesic error metrics")

    model.eval()

    # losses = []
    # geodesic_errors = []

    t_src_idx = test_dataset.get_sidx()
    name_list = test_dataset.get_names()
    # min_loss = defaultdict(int) # stores minimum L2 loss of pair model
    # res_name = defaultdict(str) # stores name of min val
    n1_idx = defaultdict(lambda: [-1, 0])
    max_diag = defaultdict(list) # stores C_pred diagonal sum and name2

    # Make torch that saves absolute diagonal sum of source model
    # sim_score = torch.zeros((args.num_test, 200-args.num_test))
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
                x.to(device) for x in shape2] # , C_gt / , C_gt.to(device)

            verts1_orig = shape1[0]
            if augment_random_rotate:
                shape1[0] = diffusion_net.utils.random_rotate_points(shape1[0])
                shape2[0] = diffusion_net.utils.random_rotate_points(shape2[0])

            # Apply the model
            C_pred, feat1, feat2 = model(shape1, shape2)
            C_pred = C_pred.squeeze(0)

            # save C_pred in psb dataset


            # Loss
            # loss = torch.mean(torch.square(C_pred-C_gt))  # L2 loss
            # losses.append(toNP(loss))

            # Update matching model
            # if min_loss[name1] == 0 or min_loss[name1] > loss:
                # min_loss[name1] = loss
                # res_name[name1] = name2
            rprob_corr, cprob_corr = diffusion_net.geometry.get_prob_fmap(C_pred)

            # Calculate Diagonal
            C_diag_r = rprob_corr.diagonal(0)
            # C_diag_c = cprob_corr.diagonal(0)
            dsum_r = C_diag_r.abs().sum() # toNP(C_diag.abs().sum())
            # dsum_c = C_diag_c.abs().sum()
            # print(dsum)
            max_diag[name1].append(name2)

            # set index for source mesh
            if n1_idx[name1] == [-1, 0]:
                n1_idx[name1][0] = cidx
                cidx += 1

            # print("{} {}".format(n1_idx[name1][0], n1_idx[name1][1]))
            # sim_score[n1_idx[name1][0], n1_idx[name1][1]] = (dsum_r + dsum_c)//2
            sim_score[n1_idx[name1][0], n1_idx[name1][1]] = dsum_r
            n1_idx[name1][1] += 1


            # # Compute the geodesic error in the vertex-to-vertex correspondence
            # if with_geodesic_error:

            #     # gather values
            #     verts1 = shape1[0]
            #     faces1 = shape1[1]
            #     evec1 = shape1[6]
            #     vts1 = shape1[10]
            #     verts2 = shape2[0]
            #     faces2 = shape2[1]
            #     evec2 = shape2[6]
            #     vts2 = shape2[10]

            #     # construct a vertex-to-vertex map via nearest neighbors from the functional map
            #     evec1_on_2 = evec1[:, :n_fmap] @ C_pred.squeeze(0).transpose(0, 1)
            #     _, pred_labels2to1 = diffusion_net.geometry.find_knn(
            #         evec2[:, :n_fmap], evec1_on_2, k=1, method='cpu_kd')
            #     pred_labels2to1 = pred_labels2to1.squeeze(-1)

            #     # measure the geodesic error for each template vertex along shape 1
            #     vts2on1 = pred_labels2to1[vts2]

            #     if not train:
            #         # print(name1)
            #         # 1-1) Save injective correspondence
            #         if fidx == 0:
            #             print("<<Test 1: Save 4 Different Shape Correspondence>>")
            #             print("1-1) Saving Injective Correspondence...")
            #         # print("<<Test 1: Save 4 Different Shape Correspondence>>")
            #         # print("1-1) Saving Injective Correspondence...")
            #         inj_label = diffusion_net.utils.toNP(pred_labels2to1)
            #         fname_lab1 = os.path.join(t_inj_path, "{0}{1}.vts".format(name1, str(fidx).zfill(3)))
            #         np.savetxt(fname_lab1, inj_label+1, fmt='%d')

            #         inj_sh1 = diffusion_net.utils.toNP(vts2on1)
            #         inj_sh2 = diffusion_net.utils.toNP(vts2)
            #         inj_gt1 = diffusion_net.utils.toNP(vts1)
            #         fname_lab1_1 = os.path.join(t_inj_path, "{0}{1}.vts".format(name1, str(fidx).zfill(3)))
            #         fname_lab1_2 = os.path.join(t_inj_path, "{0}{1}.vts".format(name2, str(fidx).zfill(3)))
            #         fname_lab1_3 = os.path.join(t_inj_path, "{0}gt1{1}.vts".format(name1, str(fidx).zfill(3)))
            #         np.savetxt(fname_lab1_1, inj_sh1+1, fmt='%d')
            #         np.savetxt(fname_lab1_2, inj_sh2+1, fmt='%d')
            #         np.savetxt(fname_lab1_3, inj_gt1+1, fmt='%d')

            #         corr_pair = diffusion_net.utils.toNP(vts2on1)
            #         fname_1 = os.path.join(t_inj_path, "tr_reg_{0}.vts".format(str(fidx).zfill(3)))
            #         np.savetxt(fname_1, corr_pair, fmt='%d')
            #         if fidx == 0:
            #             print("1-1) Injective Correspondence Saved on {}".format(t_inj_path))

            #         # 1-2) Save bijective correspondence
            #         if fidx == 0:
            #             print("1-2) Saving Bijective Correspondence...")
            #             _, bij_labels2to1 = diffusion_net.geometry.find_bij_knn(
            #                 evec2[:, :n_fmap], evec1_on_2, k=min(shape1[0].size(0), shape2[0].size(0)), method='brute')
            #             bij_labels2to1 = bij_labels2to1.squeeze(-1)
            #             bij_label = diffusion_net.utils.toNP(bij_labels2to1)
            #             fname_lab2 = os.path.join(t_bij_path, "{0}_to_{1}_label{2}.vts".format(name2, name1, str(fidx).zfill(3)))
            #             np.savetxt(fname_lab2, bij_label, fmt='%d')

            #             bij_vts2on1 = bij_labels2to1[vts2]
            #             bij_pair = diffusion_net.utils.toNP(bij_vts2on1)
            #             fname_2 = os.path.join(t_bij_path, "{0}{1}.vts".format(name1, str(fidx).zfill(3)))
            #             np.savetxt(fname_2, bij_pair+1, fmt='%d')
            #             print("1-2) Bijective Correspondence Saved on {}".format(t_bij_path))


            #     errors = diffusion_net.geometry.geodesic_label_errors(
            #         verts1_orig, faces1, vts2on1, vts1, normalization='area', geodesic_cache_dir=geodesic_cache_dir)

            #     geodesic_error = toNP(torch.mean(errors))
            #     geodesic_errors.append(geodesic_error)

            # if not train:
            #     # 1-3) Save probabilistic correspondence
            #     if fidx == 0:
            #         print("1-3) Saving Probabilistic Correspondence...")
            #     rprob_corr, cprob_corr = diffusion_net.geometry.get_prob_fmap(C_pred)
            #     C_rprob = diffusion_net.utils.toNP(rprob_corr)
            #     C_cprob = diffusion_net.utils.toNP(cprob_corr)
            #     fname_3r = os.path.join(t_prob_path, "{0}pfmap_r{1}.vts".format(name1, str(fidx).zfill(3)))
            #     fname_3c = os.path.join(t_prob_path, "{0}pfmap_c{1}.vts".format(name1, str(fidx).zfill(3)))
            #     np.savetxt(fname_3r, C_rprob, fmt='%1.3f')
            #     np.savetxt(fname_3c, C_cprob, fmt='%1.3f')
            #     if fidx == 0:
            #         print("1-3) Probabilistic Correspondence Saved on {}".format(t_prob_path))

            #     # 1-4) Save binary correspondence
            #     if fidx == 0:
            #         print("1-4) Saving Binary Correspondence...")
            #     bin_corr = diffusion_net.geometry.get_bin_fmap(C_pred)
            #     C_bin = diffusion_net.utils.toNP(bin_corr)
            #     fname_4 = os.path.join(t_bin_path, "{0}bfmap_{1}.vts".format(name1, str(fidx).zfill(3)))
            #     np.savetxt(fname_4, C_bin, fmt='%d')
            #     if fidx == 0:
            #         print("1-4) Binary Correspondence Saved on {}".format(t_bin_path))
            #         print("<<Test 1 Finished>>")
                
            #     fidx += 1

    # mean_loss = np.mean(losses)
    # mean_geodesic_error = np.mean(
    #     geodesic_errors) if with_geodesic_error else -1

    return sim_score, max_diag, n1_idx # mean_loss, min_loss, , mean_geodesic_error


# if train:
#     print("Training...")

#     for epoch in range(n_epoch):
#         train_loss = train_epoch(epoch)
#         test_loss, test_geodesic_error = test(with_geodesic_error=True)
#         print("Epoch {} - Train overall: {:.5e}  Test overall: {:.5e}  Test geodesic error: {:.5e}".format(
#             epoch, train_loss, test_loss, test_geodesic_error))

#         print(" ==> saving last model to " + model_save_path)
#         torch.save(model.state_dict(), model_save_path)


# Test
print("<<Test 3 Start>>")
sim_score, max_diag, n1_idx = test(with_geodesic_error=False) # mean_loss, pair_loss, 
# print("Overall MSE Loss: {:.4}".format(mean_loss))
_k = list(max_diag.keys())
count = 0
sim_score.to(device)
# print(sim_score)
# for i in range(args.num_test):
for i in range(len(_k)):
    score = 0
    n1 = _k[i]
    sim_idx = n1_idx[n1][0] # n1_idx가 sim_idx에서 n1이 차지하는 row, n1열에 값 넣을때 track하는 column임(200)
    top_k, ind_topk = sim_score[sim_idx].topk(args.top_k)
    top_k = toNP(top_k)
    ind_topk = toNP(ind_topk)
    # print("Test{} Source: {}".format(i+1, n1), end=' ')
    print("{}:".format(n1), end=' ')
    # print("Test{} - closest pair model: {}, loss: {}".format(i+1, pair_name[_k[i]], pair_loss[_k[i]])) # _k[i]
    # t_flag = True
    diag_n2 = max_diag[n1] # tuple of (value, name2)
    for j in range(len(top_k)): # args.top_k
        val = top_k[j]
        # find name
        n2 = diag_n2[ind_topk[j]] #[1]
        # n2 = ''
        # for k in len(diag_n2):
        #     if max_diag[n1][k][0] == val:
        #         n2 = max_diag[n][k][1]
        #         v_flag = True
        #         break
        if j == len(top_k)-1:
            # print("{}: {:.3}-{}".format(j+1, diag_n2[ind_topk[j]][0], n2))
            print("{}".format(n2))
        else:
            # print("{}: {:.3}-{}".format(j+1, diag_n2[ind_topk[j]][0], n2), end=" ")
            print("{},".format(n2), end=" ")
    # t_flag &= v_flag
    # if t_flag:
        score += diffusion_net.geometry.equal_class(n1, n2)
    if score >= args.top_k/2:
        count += 1
print("Evaluated {} out of {} tests, {}%".format(count, len(_k), round(count/(len(_k))*100)))
# print("Evaluated {} out of {} tests, {:.3}%".format(count, len(_k), count/(len(_k))*100))