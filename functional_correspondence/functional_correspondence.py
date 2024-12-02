from faust_scape_dataset import FaustScapeDataset
from diffusion_net.utils import toNP
import diffusion_net
import os
import sys
import argparse
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn
from torch.utils.data import DataLoader

from fmaps_model import FunctionalMapCorrespondenceWithDiffusionNetFeatures

# add the path to the DiffusionNet src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))


# === Options


# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true",
                    help="evaluate using the pretrained model")
parser.add_argument("--train_dataset", type=str,
                    default="faust", help="what dataset to train on")
parser.add_argument("--test_dataset", type=str,
                    default="faust", help="what dataset to test on")
parser.add_argument("--input_features", type=str,
                    help="what features to use as input ('xyz' or 'hks') default: hks", default='hks')
parser.add_argument("--load_model", type=str,
                    help="path to load a pretrained model from")
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
model_save_path = os.path.join(
    base_path, "saved_models/{}_{}.pth".format(args.train_dataset, input_features))
dataset_path = os.path.join(base_path, "data")
diffusion_net.utils.ensure_dir_exists(os.path.join(base_path, "saved_models/"))
# Test paths
test_path = os.path.join(base_path, "test", args.test_dataset)
t_inj_path = os.path.join(test_path, "inj_corres")
t_bij_path = os.path.join(test_path, "bij_corres")
t_prob_path = os.path.join(test_path, "prob_corres")
t_bin_path = os.path.join(test_path, "bin_corres")
diffusion_net.utils.ensure_dir_exists(os.path.join(base_path, "test/"))
diffusion_net.utils.ensure_dir_exists(os.path.join(test_path, "inj_corres"))
diffusion_net.utils.ensure_dir_exists(os.path.join(test_path, "bij_corres"))
diffusion_net.utils.ensure_dir_exists(os.path.join(test_path, "prob_corres"))
diffusion_net.utils.ensure_dir_exists(os.path.join(test_path, "bin_corres"))


# === Load datasets

if not args.evaluate:
    train_dataset = FaustScapeDataset(dataset_path, name=args.train_dataset, train=True,
                                      k_eig=k_eig, n_fmap=n_fmap, use_cache=True, op_cache_dir=op_cache_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=None, shuffle=True)

else:
    test_dataset = FaustScapeDataset(dataset_path, name=args.test_dataset, train=False,
                                    k_eig=k_eig, n_fmap=n_fmap, use_cache=True, op_cache_dir=op_cache_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=None, shuffle=False)


# === Create the model

C_in = {'xyz': 3, 'hks': 16}[input_features]  # dimension of input features

model = FunctionalMapCorrespondenceWithDiffusionNetFeatures(
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


def train_epoch(epoch):

    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()

    losses = []

    for data in tqdm(train_loader):

        # Get data
        shape1, shape2, C_gt = data
        *shape1, name1 = shape1
        *shape2, name2 = shape2
        shape1, shape2, C_gt = [x.to(device) for x in shape1], [x.to(
            device) for x in shape2], C_gt.to(device).unsqueeze(0)

        # Randomly rotate positions
        if augment_random_rotate:
            shape1[0] = diffusion_net.utils.random_rotate_points(shape1[0])
            shape2[0] = diffusion_net.utils.random_rotate_points(shape2[0])

        # Apply the model
        C_pred, feat1, feat2 = model(shape1, shape2)

        # Evaluate loss
        loss = torch.mean(torch.square(C_pred-C_gt))  # L2 loss
        losses.append(toNP(loss))
        loss.backward()

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_loss = np.mean(losses)

    return train_loss


# Do an evaluation pass on the test dataset
def test(with_geodesic_error=False):

    if with_geodesic_error:
        print("Evaluating geodesic error metrics")

    model.eval()

    losses = []
    geodesic_errors = []

    with torch.no_grad():

        fidx = 0
        for data in tqdm(test_loader):

            # Get data
            shape1, shape2, C_gt = data
            *shape1, name1 = shape1
            *shape2, name2 = shape2
            shape1, shape2, C_gt = [x.to(device) for x in shape1], [
                x.to(device) for x in shape2], C_gt.to(device)

            verts1_orig = shape1[0]
            if augment_random_rotate:
                shape1[0] = diffusion_net.utils.random_rotate_points(shape1[0])
                shape2[0] = diffusion_net.utils.random_rotate_points(shape2[0])

            # Apply the model
            C_pred, feat1, feat2 = model(shape1, shape2)
            C_pred = C_pred.squeeze(0)

            # Loss
            loss = torch.mean(torch.square(C_pred-C_gt))  # L2 loss
            losses.append(toNP(loss))

            # Compute the geodesic error in the vertex-to-vertex correspondence
            if with_geodesic_error:

                # gather values
                verts1 = shape1[0]
                faces1 = shape1[1]
                evec1 = shape1[6]
                vts1 = shape1[10]
                verts2 = shape2[0]
                faces2 = shape2[1]
                evec2 = shape2[6]
                vts2 = shape2[10]


                # construct a vertex-to-vertex map via nearest neighbors from the functional map
                evec1_on_2 = evec1[:, :n_fmap] @ C_pred.squeeze(0).transpose(0, 1)
                _, pred_labels2to1 = diffusion_net.geometry.find_knn(
                    evec2[:, :n_fmap], evec1_on_2, k=1, method='cpu_kd')
                pred_labels2to1 = pred_labels2to1.squeeze(-1)

                # measure the geodesic error for each template vertex along shape 1
                # for i in range(len(pred_labels2to1)):
                vts2on1 = pred_labels2to1[vts2]

                if not train:
                    # print(name1)
                    # 1-1) Save injective correspondence
                    if fidx == 0:
                        print("<<Test 1: Save 4 Different Shape Correspondence>>")
                        print("1-1) Saving Injective Correspondence...")
                        # print(vts2.size())
                    # print("<<Test 1: Save 4 Different Shape Correspondence>>")
                    # print("1-1) Saving Injective Correspondence...")
                    inj_label = diffusion_net.utils.toNP(pred_labels2to1)
                    fname_lab1 = os.path.join(t_inj_path, "{0}{1}.vts".format(name1, str(fidx).zfill(3)))
                    np.savetxt(fname_lab1, inj_label+1, fmt='%d')

                    inj_sh1 = diffusion_net.utils.toNP(vts2on1)
                    inj_sh2 = diffusion_net.utils.toNP(vts2)
                    inj_gt1 = diffusion_net.utils.toNP(vts1)
                    fname_lab1_1 = os.path.join(t_inj_path, "{0}{1}.vts".format(name1, str(fidx).zfill(3)))
                    fname_lab1_2 = os.path.join(t_inj_path, "{0}{1}.vts".format(name2, str(fidx).zfill(3)))
                    fname_lab1_3 = os.path.join(t_inj_path, "{0}gt1{1}.vts".format(name1, str(fidx).zfill(3)))
                    np.savetxt(fname_lab1_1, inj_sh1+1, fmt='%d')
                    np.savetxt(fname_lab1_2, inj_sh2+1, fmt='%d')
                    np.savetxt(fname_lab1_3, inj_gt1+1, fmt='%d')

                    corr_pair = diffusion_net.utils.toNP(vts2on1)
                    fname_1 = os.path.join(t_inj_path, "tr_reg_{0}.vts".format(str(fidx).zfill(3)))
                    np.savetxt(fname_1, corr_pair, fmt='%d')
                    if fidx == 0:
                        print("1-1) Injective Correspondence Saved on {}".format(t_inj_path))

                    # 1-2) Save bijective correspondence
                    if fidx == 0:
                        print("1-2) Saving Bijective Correspondence...")
                        # print(shape1[0].size(0))
                        # print(shape2[0].size(0))
                        lab_len = min(shape1[0].size(0), shape2[0].size(0))
                        _, bij_labels2to1 = diffusion_net.geometry.find_bij_knn(
                            evec2[:, :n_fmap], evec1_on_2, k=lab_len, method='brute')
                        bij_labels2to1 = bij_labels2to1.squeeze(-1)
                        bij_label = diffusion_net.utils.toNP(bij_labels2to1)
                        fname_lab2 = os.path.join(t_bij_path, "{0}_to_{1}_label{2}.vts".format(name2, name1, str(fidx).zfill(3)))
                        np.savetxt(fname_lab2, bij_label, fmt='%d')

                        bij_vts2 = vts2.squeeze(-1)
                        bij_vts2 = bij_vts2.clamp(min=0, max=lab_len-1)
                        bij_vts2 = bij_vts2.unsqueeze(1)
                        bij_vts2on1 = bij_labels2to1[bij_vts2]
                        bij_pair = diffusion_net.utils.toNP(bij_vts2on1)
                        fname_2 = os.path.join(t_bij_path, "{0}{1}.vts".format(name1, str(fidx).zfill(3)))
                        np.savetxt(fname_2, bij_pair+1, fmt='%d')
                        print("1-2) Bijective Correspondence Saved on {}".format(t_bij_path))


                errors = diffusion_net.geometry.geodesic_label_errors(
                    verts1_orig, faces1, vts2on1, vts1, normalization='area', geodesic_cache_dir=geodesic_cache_dir)

                geodesic_error = toNP(torch.mean(errors))
                geodesic_errors.append(geodesic_error)

            if not train:
                # 1-3) Save probabilistic correspondence
                if fidx == 0:
                    print("1-3) Saving Probabilistic Correspondence...")
                rprob_corr, cprob_corr = diffusion_net.geometry.get_prob_fmap(C_pred)
                C_rprob = diffusion_net.utils.toNP(rprob_corr)
                C_cprob = diffusion_net.utils.toNP(cprob_corr)
                fname_3r = os.path.join(t_prob_path, "{0}pfmap_r{1}.vts".format(name1, str(fidx).zfill(3)))
                fname_3c = os.path.join(t_prob_path, "{0}pfmap_c{1}.vts".format(name1, str(fidx).zfill(3)))
                np.savetxt(fname_3r, C_rprob, fmt='%1.3f')
                np.savetxt(fname_3c, C_cprob, fmt='%1.3f')
                if fidx == 0:
                    print("1-3) Probabilistic Correspondence Saved on {}".format(t_prob_path))

                # 1-4) Save binary correspondence
                if fidx == 0:
                    print("1-4) Saving Binary Correspondence...")
                bin_corr = diffusion_net.geometry.get_bin_fmap(C_pred)
                C_bin = diffusion_net.utils.toNP(bin_corr)
                fname_4 = os.path.join(t_bin_path, "{0}bfmap_{1}.vts".format(name1, str(fidx).zfill(3)))
                np.savetxt(fname_4, C_bin, fmt='%d')
                if fidx == 0:
                    print("1-4) Binary Correspondence Saved on {}".format(t_bin_path))
                    print("<<Test 1 Finished>>")
                
                fidx += 1

    mean_loss = np.mean(losses)
    mean_geodesic_error = np.mean(
        geodesic_errors) if with_geodesic_error else -1

    return mean_loss, mean_geodesic_error


if train:
    print("Training...")

    for epoch in range(n_epoch):
        train_loss = train_epoch(epoch)
        test_loss, test_geodesic_error = test(with_geodesic_error=True)
        print("Epoch {} - Train overall: {:.5e}  Test overall: {:.5e}  Test geodesic error: {:.5e}".format(
            epoch, train_loss, test_loss, test_geodesic_error))

        print(" ==> saving last model to " + model_save_path)
        torch.save(model.state_dict(), model_save_path)


# Test
mean_loss, mean_geodesic_error = test(with_geodesic_error=True)
print("<<Test 2: Shape Correspondence Geodesic Error: below 5%>>")
print("Overall MSE Loss: {:.4},  Geodesic error (%): {:.4}%".format(
    mean_loss, mean_geodesic_error*100))
if mean_geodesic_error*100 < 5:
    print("PASS")
else:
    print("FAIL")
print("<<Test 2 Finished>>")
# print("<<Test 3 Start>>")

