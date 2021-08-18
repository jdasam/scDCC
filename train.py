from time import time
import os

import torch
import anndata as ad
from scDCC import scDCC
import numpy as np
from sklearn import metrics
import h5py
import scanpy.api as sc
from preprocess import read_dataset, normalize
from utils import cluster_acc
from pathlib import Path
import argparse


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=31, type=int)
    parser.add_argument('--label_cells', default=0.1, type=float)
    parser.add_argument('--label_cells_files', default='label_selected_cells_1.txt')
    parser.add_argument('--n_pairwise', default=20000, type=int)
    parser.add_argument('--n_pairwise_error', default=0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='../data/CITE_PBMC/10X_PBMC_select_2100.h5')
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--ml_weight', default=1., type=float,
                        help='coefficient of must-link loss')
    parser.add_argument('--cl_weight', default=1., type=float,
                        help='coefficient of cannot-link loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default=Path('results/test2/'), type=Path)
    parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')
    

    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # adata = ad.read_h5ad("/home/svcapp/tbrain_x/azimuth/pbmc_multimodal_raw.h5ad")
    # adata.obs = ad.read_h5ad("/home/svcapp/tbrain_x/azimuth/pbmc_multimodal.h5ad").obs

    # cell_type_names, y = np.unique(adata.obs["celltype.l2"].values, return_inverse=True)

    # sc.pp.filter_genes(adata, min_counts=20000)
    # sc.pp.filter_cells(adata, min_counts=100)
    # adata.raw = adata.copy()
    # sc.pp.normalize_per_cell(adata)
    # adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    # sc.pp.log1p(adata)
    # sc.pp.scale(adata)
    # adata.write_h5ad("pbmc_normalized.h5ad")
    '''
    adata = ad.read_h5ad("pbmc_normalized.h5ad")
    cell_type_names, y = np.unique(adata.obs["celltype.l2"].values, return_inverse=True)
    '''

    if args.data_file.endswith('h5ad'):
        adata = ad.read_h5ad(args.data_file)
        cell_type_names, y = np.unique(adata.obs["celltype.l2"].values, return_inverse=True)
    elif args.data_file.endswith('h5'):
        data_mat = h5py.File(args.data_file)
        x = np.array(data_mat['X'])
        y = np.array(data_mat['Y'])
        protein_markers = np.array(data_mat['ADT_X'])
        data_mat.close()
        adata = sc.AnnData(x)
        adata.obs['Group'] = y
        adata = read_dataset(adata,
                        transpose=False,
                        test_split=False,
                        copy=True)
        adata = normalize(adata,
                        size_factors=True,
                        normalize_input=True,
                        logtrans_input=True)
    else:
        raise Exception("Undefined data type")

    input_size = adata.n_vars
    if not os.path.exists(args.label_cells_files):
        indx = np.arange(len(y))
        np.random.shuffle(indx)
        label_cell_indx = indx[0:int(np.ceil(args.label_cells*len(y)))]
    else:
        label_cell_indx = np.loadtxt(args.label_cells_files, dtype=np.int)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

    # ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair_by_label(y,label_cell_indx, args.n_pairwise)

    # print("Must link paris: %d" % ml_ind1.shape[0])
    # print("Cannot link paris: %d" % cl_ind1.shape[0])

    sd = 2.5

    model = scDCC(input_dim=adata.n_vars, z_dim=32, n_clusters=args.n_clusters, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=args.gamma,
                ml_weight=args.ml_weight, cl_weight=args.ml_weight, save_dir=args.save_dir).cuda()
    
    print(str(model))

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                                batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    print('Pretraining time: %d seconds.' % int(time() - t0))


    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, num_links=args.n_pairwise, y=y, batch_size=args.batch_size, num_epochs=args.maxiter, 
                # ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    print('Total time: %d seconds.' % int(time() - t0))

    eval_cell_y_pred = np.delete(y_pred, label_cell_indx)
    eval_cell_y = np.delete(y, label_cell_indx)
    acc = np.round(cluster_acc(eval_cell_y, eval_cell_y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(eval_cell_y, eval_cell_y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(eval_cell_y, eval_cell_y_pred), 5)
    print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

    if not os.path.exists(args.label_cells_files):
        np.savetxt(args.label_cells_files, label_cell_indx, fmt="%i")