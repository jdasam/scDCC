from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from layers import ZINBLoss, MeanAct, DispAct
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
from utils import cluster_acc

from typing import Callable, Any, Union
from scipy import sparse
import random
from torch.utils.tensorboard import SummaryWriter

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class scDCC(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, save_dir, encodeLayer=[], decodeLayer=[], 
            activation="relu", sigma=1., alpha=1., gamma=1., ml_weight=1., cl_weight=1., tb=True):
        super(scDCC, self).__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        self.save_dir = str(save_dir)
        self.tb_writer = SummaryWriter(log_dir=save_dir) if tb else None

        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))
        self.zinb_loss = ZINBLoss().cuda()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
    
    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
    
    def forward(self, x):
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi
    
    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []

        dataset = SingleCellOnlyX(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        # num = X.shape[0]
        # num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        # for batch_idx in range(num_batch):
        for batch in dataloader:
            # xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            # inputs = Variable(xbatch)
            if use_cuda:
                batch = batch.cuda()
            z,_, _, _, _ = self.forward(batch)
            encoded.append(z.data)
        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return self.gamma*kldloss

    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return self.ml_weight*ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return self.cl_weight*cl_loss

    def pretrain_autoencoder(self, x, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        # dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataset = SingleCellRecon(x, X_raw, size_factor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        itr= 0
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).cuda()
                x_raw_tensor = Variable(x_raw_batch).cuda()
                sf_tensor = Variable(sf_batch).cuda()
                _, _, mean_tensor, disp_tensor, pi_tensor = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 3)
                optimizer.step()
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
                if self.tb_writer:
                    self.tb_writer.add_scalar('Loss/pretrain_ZINB', loss.item(), itr)
                itr+=1
        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, sf, num_links,
            ml_p=1., cl_p=1., y=None, lr=1., batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''
        X:          tensor data
        X_raw:      raw data for calculating reconstruction loss 
        sf:         size fztor
        ml_ind1:    
        ml_ind2:    
        cl_ind1:    cannot link indice start?
        cl_ind2:    cannot link indice end?
        y:          ground truth label
        ml_p: must link weight?
        cl_p: cannot link weight?
        '''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Clustering stage")
        # X = torch.tensor(X).cuda()
        # X_raw = torch.tensor(X_raw).cuda()
        # sf = torch.tensor(sf).cuda()
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20)
        data = self.encodeBatch(X)
        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
        
        self.train()
        num = X.shape[0]
        # num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        # ml_num_batch = int(math.ceil(1.0*ml_ind1.shape[0]/batch_size))
        # cl_num_batch = int(math.ceil(1.0*cl_ind1.shape[0]/batch_size))
        # cl_num = cl_ind1.shape[0]
        # ml_num = ml_ind1.shape[0]

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        update_ml = 1
        update_cl = 1

        dataset = SingleCellRecon(X, X_raw, sf)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        itr = 0
        link_itr = 0
        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the target distribution p
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data
                dataset.p = p

                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch+1, acc, nmi, ari))
                    if self.tb_writer:
                        self.tb_writer.add_scalar('Cluster/Acc', acc, epoch)
                        self.tb_writer.add_scalar('Cluster/NMI', nmi, epoch)
                        self.tb_writer.add_scalar('Cluster/ARI', ari, epoch)

                # save current model
                if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                    self.save_checkpoint({'epoch': epoch+1,
                            'state_dict': self.state_dict(),
                            'mu': self.mu,
                            'p': p,
                            'q': q,
                            'y_pred': self.y_pred,
                            'y_pred_last': self.y_pred_last,
                            'y': y
                            }, epoch+1, filename=save_dir)

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            for batch in dataloader:
                xbatch, xrawbatch, sfbatch, pbatch = batch
            # for batch_idx in range(num_batch):
                # xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                # xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                # sfbatch = sf[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch).cuda()
                rawinputs = Variable(xrawbatch).cuda()
                sfinputs = Variable(sfbatch).cuda()
                target = Variable(pbatch).cuda()

                z, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                loss = cluster_loss + recon_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs)
                recon_loss_val += recon_loss.data * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val
                if self.tb_writer:
                    self.tb_writer.add_scalar('Loss/cluster_loss', cluster_loss, itr)
                    self.tb_writer.add_scalar('Loss/recon_loss', recon_loss, itr)
                itr += 1

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f ZINB Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))

            link_dataset = SingleCellLink(X, X_raw, sf, y, num_links)
            link_dataloader = DataLoader(link_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            ml_loss = 0.0
            cl_loss = 0.0
            if epoch % update_ml == 0:
                for batch in link_dataloader:
                    link_x, link_xraw, link_sf = batch
                    link_xraw = link_xraw.cuda()
                    link_sf = link_sf.float().cuda()
                    optimizer.zero_grad()
                    z, q, mean, disp, pi = self.forward(link_x.view(-1, link_x.shape[-1]).cuda())
                    # print(z.shape, q.shape, mean.shape, disp.shape, pi.shape)
                    z, q, mean, disp, pi = reshape_by_batch([z, q, mean, disp, pi], link_x.shape[0], link_x.shape[1])
                    z, q, mean, disp, pi, link_xraw = [chunk_and_squeeze(x) for x in (z, q, mean, disp, pi, link_xraw)]

                    # z2, q2, mean2, disp2, pi2 = self.forward(inputs2)
                    temp_ml_loss = self.pairwise_loss(q[0], q[1], "ML")
                    temp_cl_loss = self.pairwise_loss(q[2], q[3], "CL")
                    loss = (ml_p*temp_ml_loss + cl_p*temp_cl_loss +
                            self.zinb_loss(link_xraw[0], mean[0], disp[0], pi[0], link_sf[:,0]) + self.zinb_loss(link_xraw[1], mean[1], disp[1], pi[1], link_sf[:,1]))
                    # 0.1 for mnist/reuters, 1 for fashion, the parameters are tuned via grid search on validation set
                    ml_loss += temp_ml_loss.data
                    cl_loss += temp_cl_loss.data
                    loss.backward()
                    optimizer.step()
                    if self.tb_writer:
                        self.tb_writer.add_scalar('Loss/must_link_loss', temp_ml_loss.item(), link_itr)
                        self.tb_writer.add_scalar('Loss/cannot_link_loss', temp_cl_loss.item(), link_itr)
                    link_itr += 1
            # cl_loss = 0.0
            # if epoch % update_cl == 0:
            #     for cl_batch_idx in range(cl_num_batch):
            #         px1 = X[cl_ind1[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
            #         px2 = X[cl_ind2[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
            #         optimizer.zero_grad()
            #         inputs1 = Variable(px1)
            #         inputs2 = Variable(px2)
            #         z1, q1, _, _, _ = self.forward(inputs1)
            #         z2, q2, _, _, _ = self.forward(inputs2)
            #         loss = cl_p*self.pairwise_loss(q1, q2, "CL")
            #         cl_loss += loss.data
            #         loss.backward()
            #         optimizer.step()

            # if ml_num_batch >0 and cl_num_batch > 0:
            # print("Pairwise Total:", round(float(ml_loss.cpu()), 2) + float(cl_loss.cpu()), "ML loss", float(ml_loss.cpu()), "CL loss:", float(cl_loss.cpu()))
            print("Pairwise Total:", float(ml_loss.cpu()) + float(cl_loss.cpu()), "ML loss", float(ml_loss.cpu()), "CL loss:", float(cl_loss.cpu()))
            if self.tb_writer:
                self.tb_writer.add_scalar('Loss/pairwise_total', (ml_loss+cl_loss).cpu().item(), epoch)
                self.tb_writer.add_scalar('Loss/must_link_total', ml_loss.cpu().item(), epoch)
                self.tb_writer.add_scalar('Loss/cannot_link_total', cl_loss.cpu().item(), epoch)

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch

def chunk_and_squeeze(tensor, n_chunks=4, dim=1):
    return [x.squeeze() for x in torch.chunk(tensor, n_chunks, dim=dim)]

def reshape_by_batch(data_list, batch_size, num_mid=4):
    return [x.view(batch_size, num_mid, x.shape[-1]) for x in data_list]

class SingleCellOnlyX(Dataset):
    def __init__(
        self,
        X: Union[sparse.csr.csr_matrix, np.ndarray],
    ) -> None:
        '''
        Load single cell expression profiles.

        Parameters
        ----------
        dataset = TensorDataset(torch.Tensor(x))

        X : np.ndarray, sparse.csr_matrix
            [Cells, Genes] expression count matrix.
        Returns
        -------
        None.
        '''
        super(SingleCellOnlyX, self).__init__()
        
        # check types on input arrays
        if type(X) not in (np.ndarray, sparse.csr_matrix,):
            msg = f'X is type {type(X)}, must `np.ndarray` or `sparse.csr_matrix`'
            raise TypeError(msg)
        
        self.X = X

    def __len__(self,) -> int:
        '''Return the number of examples in the data set.'''
        return self.X.shape[0]

    def __getitem__(self, idx: int,) -> dict:
        '''Get a single cell expression profile and corresponding label.

        Parameters
        ----------
        idx : int
            index value in `range(len(self))`.

        Returns
        -------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))

        '''
        if type(idx) != int:
            raise TypeError(
                f'indices must be int, you passed {type(idx)}, {idx}'
            )
        
        # check if the idx value is valid given the dataset size
        if idx < 0 or idx > len(self):
            vals = (idx, len(self))
            raise ValueError(
                'idx %d is invalid for dataset with %d examples.' % vals)

        # retrieve relevant sample vector and associated label
        # store in a hash table for later manipulation and retrieval
        
        # input_ is either an `np.ndarray` or `sparse.csr.csr_matrix`
        input_ = self.X[idx, ...]
        
        # if the corresponding vectors are sparse, convert them to dense
        # we perform this operation on a samplewise-basis to avoid
        # storing the whole count matrix in dense format
        if type(input_) != np.ndarray:
            input_ = input_.toarray()
            
        input_ = torch.from_numpy(input_).float()
        if input_.size(0) == 1:
            input_ = input_.squeeze()
        return input_


class SingleCellRecon(Dataset):
    '''Dataset class for loading single cell profiles.

    Attributes
    ----------
    X : np.ndarray, sparse.csr_matrix
        [Cells, Genes] cell profiles.
    y_labels : np.ndarray, sparse.csr_matrix
        [Cells,] integer class labels.
    y : torch.FloatTensor
        [Cells, Classes] one hot labels.
    transform : Callable
        performs data transformation operations on a
        `sample` dict.
    num_classes : int
        number of classes in the dataset. default `-1` infers
        the number of classes as `len(unique(y))`.
    '''

    def __init__(
        self,
        X: Union[sparse.csr.csr_matrix, np.ndarray],
        X_raw: Union[sparse.csr.csr_matrix, np.ndarray],
        size_factor,
    ) -> None:
        '''
        Load single cell expression profiles.

        Parameters
        ----------
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))

        X : np.ndarray, sparse.csr_matrix
            [Cells, Genes] expression count matrix.
            scNym tools expect ln(Counts Per Million + 1).
        X : np.ndarray, sparse.csr_matrix
            [Cells, Genes] expression count matrix.
            scNym tools expect ln(Counts Per Million + 1).

        Returns
        -------
        None.
        '''
        super(SingleCellRecon, self).__init__()
        
        # check types on input arrays
        if type(X) not in (np.ndarray, sparse.csr_matrix,):
            msg = f'X is type {type(X)}, must `np.ndarray` or `sparse.csr_matrix`'
            raise TypeError(msg)

        if type(X_raw) not in (np.ndarray, sparse.csr_matrix,):
            msg = f'X is type {type(X_raw)}, must `np.ndarray` or `sparse.csr_matrix`'
            raise TypeError(msg)
        
        self.X = X
        self.X_raw = X_raw
        self.size_factor = size_factor

    def __len__(self,) -> int:
        '''Return the number of examples in the data set.'''
        return self.X.shape[0]

    def __getitem__(self, idx: int,) -> dict:
        '''Get a single cell expression profile and corresponding label.

        Parameters
        ----------
        idx : int
            index value in `range(len(self))`.

        Returns
        -------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))

        '''
        if type(idx) != int:
            raise TypeError(
                f'indices must be int, you passed {type(idx)}, {idx}'
            )
        
        # check if the idx value is valid given the dataset size
        if idx < 0 or idx > len(self):
            vals = (idx, len(self))
            raise ValueError(
                'idx %d is invalid for dataset with %d examples.' % vals)

        # retrieve relevant sample vector and associated label
        # store in a hash table for later manipulation and retrieval
        
        # input_ is either an `np.ndarray` or `sparse.csr.csr_matrix`
        input_ = self.X[idx, ...]
        input_raw = self.X[idx, ...]
        
        # if the corresponding vectors are sparse, convert them to dense
        # we perform this operation on a samplewise-basis to avoid
        # storing the whole count matrix in dense format
        if type(input_) != np.ndarray:
            input_ = input_.toarray()
            input_raw = input_raw.toarray()
            
        input_ = torch.from_numpy(input_).float()
        input_raw = torch.from_numpy(input_raw).float()
        if input_.size(0) == 1:
            input_ = input_.squeeze()
            input_raw = input_raw.squeeze()
        
        if hasattr(self, "p"):
            return input_, input_raw, self.size_factor[idx], self.p[idx]
        return input_, input_raw, self.size_factor[idx]

class SingleCellLink(SingleCellRecon):
    def __init__(self, 
            X: Union[sparse.csr.csr_matrix, np.ndarray], 
            X_raw: Union[sparse.csr.csr_matrix, np.ndarray], 
            size_factor,
            class_labels,
            num_links: int) -> None:
        '''
        links: Tuple of (must_link_start, must_link_end, cannot_link_start, cannot_link_end)
        '''
        super().__init__(X, X_raw, size_factor)
        self.num_links = num_links
        self.class_labels = class_labels

        self.candidates = list(range(len(class_labels)))

        self.index_by_class = {x:np.where(class_labels==x)[0].tolist() for x in set(class_labels)}
        
    def __len__(self):
        return self.num_links

    def get_value(self, data, idx):
        input_ = data[idx, ...]
        if type(input_) != np.ndarray:
            input_ = input_.toarray()
        input_ = torch.from_numpy(input_).float()
        if input_.size(0) == 1:
            input_ = input_.squeeze()
        return input_

    def get_value_by_ids(self, data, idxs):
        return torch.stack([self.get_value(data, x) for x in idxs])

    def __getitem__(self, index):
        # sampled_class = random.randint(len(self.index_by_class)-1)
        # sampled_index = random.sample(self.index_by_class[sampled_class], 2)
        ml_idx1, cl_idx1, cl_idx2 = random.sample(self.candidates, 3)
        ml_idx1_cls = self.class_labels[ml_idx1]
        ml_idx2= random.sample(self.index_by_class[ml_idx1_cls], 1)[0]
        while ml_idx1 == ml_idx2:
            ml_idx2= random.sample(self.index_by_class[ml_idx1_cls], 1)[0]
        cl_idx1_cls = self.class_labels[cl_idx1]
        cl_idx2_cls = self.class_labels[cl_idx2]
        while cl_idx1_cls == cl_idx2_cls:
            cl_idx2= random.sample(self.candidates, 1)[0]
            cl_idx2_cls = self.class_labels[cl_idx2]
        
        ids = [ml_idx1, ml_idx2, cl_idx1, cl_idx2]
    
        return self.get_value_by_ids(self.X, ids), self.get_value_by_ids(self.X_raw, ids), self.size_factor[ids].values
        
