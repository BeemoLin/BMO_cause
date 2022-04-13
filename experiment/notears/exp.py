#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class NotearsSobolev(nn.Module):
    def __init__(self, d, k):
        """d: num variables k: num expansion of each variable"""
        super(NotearsSobolev, self).__init__()
        self.d, self.k = d, k
        self.fc1_pos = nn.Linear(d * k, d, bias=False)  # ik -> j
        self.fc1_neg = nn.Linear(d * k, d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self):
        # weight shape [j, ik]
        bounds = []
        for j in range(self.d):
            for i in range(self.d):
                for _ in range(self.k):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def sobolev_basis(self, x):  # [n, d] -> [n, dk]
        seq = []
        for kk in range(self.k):
            mu = 2.0 / (2 * kk + 1) / math.pi  # sobolev basis
            psi = mu * torch.sin(x / mu)
            seq.append(psi)  # [n, d] * k
        bases = torch.stack(seq, dim=2)  # [n, d, k]
        bases = bases.view(-1, self.d * self.k)  # [n, dk]
        return bases

    def forward(self, x):  # [n, d] -> [n, d]
        bases = self.sobolev_basis(x)  # [n, dk]
        x = self.fc1_pos(bases) - self.fc1_neg(bases)  # [n, d]
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x

    def h_func(self):
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(self.d) + A / self.d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, self.d - 1)
        # h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self):
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# def main():
#     torch.set_default_dtype(torch.double)
#     np.set_printoptions(precision=3)

#     import notears.utils as ut
#     ut.set_random_seed(123)

#     n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
#     B_true = ut.simulate_dag(d, s0, graph_type)
#     np.savetxt('W_true.csv', B_true, delimiter=',')

#     X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
#     np.savetxt('X.csv', X, delimiter=',')

#     model = NotearsMLP(dims=[d, 10, 1], bias=True)
#     W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
#     assert ut.is_dag(W_est)
#     np.savetxt('W_est.csv', W_est, delimiter=',')
#     acc = ut.count_accuracy(B_true, W_est != 0)
#     print(acc)


# if __name__ == '__main__':
#     main()


# In[ ]:


def experiment(exp_name='n_500_d_20_e_12_ER'):

    from notears.locally_connected import LocallyConnected
    from notears.lbfgsb_scipy import LBFGSBScipy
    from notears.trace_expm import trace_expm
    import torch
    import torch.nn as nn
    import numpy as np
    import math

    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import notears.utils as ut
    ut.set_random_seed(123)



    import os
    import logging
    from pytz import timezone
    from datetime import datetime

    from helpers.log_helper import LogHelper
    from helpers.dir_utils import create_dir
    from helpers.analyze_utils import count_accuracy, plot_estimated_graph

    import time

    start = time.time()

    # In[ ]:


    import numpy as np
    import pandas as pd
    import graphviz
    import lingam
    from lingam.utils import make_dot

    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])

    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(1)


    dataset_path = '../../experiment/datasets/{}/X.csv'.format(exp_name)
    true_path = '../../experiment/datasets/{}/W_true.csv'.format(exp_name)

    df = pd.read_csv(dataset_path, header=None)
    X = pd.DataFrame(df)

    model = lingam.DirectLiNGAM()
    model.fit(X)

    W_est = model.adjacency_matrix_

    W_true = pd.read_csv(true_path, header=None)
    
    # Setup for logging
    output_dir = 'output/{}/{}'.format(exp_name, datetime.now(timezone('Asia/Taipei')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir), level_str='INFO')
    _logger = logging.getLogger(__name__)
    
    # Save w_est png
    dot = make_dot(W_est)

    dot.format = 'png'

    w_est_dag = dot.render('{}/w_est_dag'.format(output_dir))

    # Save w_true png
    dot = make_dot(W_true)

    dot.format = 'png'

    w_true_dag = dot.render('{}/w_true_dag'.format(output_dir))
    
    model.adjacency_matrix_


    # Plot raw estimated graph
    plot_estimated_graph(W_est, W_true,
                         save_name='{}/raw_estimated_graph.png'.format(output_dir))

    _logger.info('Thresholding.')
    # Plot thresholded estimated graph
    graph_thres = 0
    copy_W_est = W_est
    copy_W_est[np.abs(W_est) < graph_thres] = 0   # Thresholding
    plot_estimated_graph(copy_W_est, W_true,
                         save_name='{}/thresholded_estimated_graph.png'.format(output_dir))
    results_thresholded = count_accuracy(W_true, W_est)
    _logger.info('Results after thresholding by {}: {}'.format(graph_thres, results_thresholded))

    end = time.time()

    _logger.info('The time used to execute this is given below')
    _logger.info(end - start)

    # draw est dag
    from IPython.display import Image
    Image(w_est_dag) 

    # draw true dag
    from IPython.display import Image
    Image(w_true_dag) 

# experiment()


# In[ ]:


# exp_name = [
#     'n_500_d_20_e_12_ER',
#     'n_500_d_20_e_12_SF',
#     'n_500_d_100_e_60_ER',
#     'n_500_d_100_e_60_SF',
    
#     'n_2000_d_20_e_12_ER',
#     'n_2000_d_20_e_12_SF',
#     'n_2000_d_100_e_60_ER',
#     'n_2000_d_100_e_60_SF']


# In[ ]:


exp_name = [
    
    'n_500_d_300_e_180_ER',
    'n_500_d_300_e_180_SF',
    'n_2000_d_300_e_180_ER',
    'n_2000_d_300_e_180_SF']

for e_name in exp_name:
    print(e_name)
    experiment(e_name)

