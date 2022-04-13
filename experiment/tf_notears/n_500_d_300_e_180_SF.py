#!/usr/bin/env python
# coding: utf-8

# In[1]:

import time

start = time.time()

import os
import logging
from pytz import timezone
from datetime import datetime
import numpy as np

from data_loader import SyntheticDataset
from models import NoTears
from trainers import ALTrainer
from helpers.config_utils import save_yaml_config, get_train_args
from helpers.log_helper import LogHelper
from helpers.tf_utils import set_seed
from helpers.dir_utils import create_dir
from helpers.analyze_utils import count_accuracy, plot_estimated_graph

# For logging of tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[2]:


import pandas as pd


#exp_name = 'n_500_d_5_e_3_ER'

# exp_name = 'n_500_d_20_e_12_ER'
# exp_name = 'n_500_d_20_e_12_SF'
# exp_name = 'n_500_d_100_e_60_ER'
# exp_name = 'n_500_d_100_e_60_SF'
# exp_name = 'n_500_d_300_e_180_ER'
exp_name = 'n_500_d_300_e_180_SF'

# exp_name = 'n_2000_d_20_e_12_ER'
# exp_name = 'n_2000_d_20_e_12_SF'
# exp_name = 'n_2000_d_100_e_60_ER'
# exp_name = 'n_2000_d_100_e_60_SF'
# exp_name = 'n_2000_d_300_e_180_ER'
# exp_name = 'n_2000_d_300_e_180_SF'

dataset_path = '../../experiment/datasets/{}/X.csv'.format(exp_name)
true_path = '../../experiment/datasets/{}/W_true.csv'.format(exp_name)

# headers = pd.read_csv(dataset_path, nrows=0).columns.tolist()
# headers = [c for c in headers if c != 'Samples']
# df = pd.read_csv(dataset_path, usecols=headers)

df = pd.read_csv(dataset_path, header=None)
X = df.to_numpy()

headers = []
for i in range(X.shape[1]):
    headers.append('x{}'.format(i))

X.shape


# In[3]:


headers


# In[4]:


w_true = pd.read_csv(true_path, header=None)


# In[5]:


df


# In[6]:


# Get arguments parsed
args = get_train_args()

args.n = X.shape[0]
args.d = X.shape[1]


# In[7]:


# Setup for logging
output_dir = 'output/{}/{}'.format(exp_name, datetime.now(timezone('Asia/Taipei')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
create_dir(output_dir)
LogHelper.setup(log_path='{}/training.log'.format(output_dir), level_str='INFO')
_logger = logging.getLogger(__name__)

# Save the configuration for logging purpose
save_yaml_config(args, path='{}/config.yaml'.format(output_dir))

# Reproducibility
set_seed(args.seed)

# Get dataset
dataset = SyntheticDataset(args.n, args.d, args.graph_type, args.degree, args.sem_type,
                           args.noise_scale, args.dataset_type)
_logger.info('Finished generating dataset')


# In[8]:


dataset.X = X
X


# In[9]:


# true answer
# dataset.W = np.zeros(shape=(args.d, args.d))
dataset.W = w_true.to_numpy()
w_true


# In[10]:


model = NoTears(args.n, args.d, args.seed, args.l1_lambda, args.use_float64)
model.print_summary(print_func=model.logger.info)

trainer = ALTrainer(args.init_rho, args.rho_max, args.h_factor, args.rho_multiply,
                    args.init_iter, args.learning_rate, args.h_tol)
W_est = trainer.train(model, dataset.X, dataset.W, args.graph_thres,
                      args.max_iter, args.iter_step, output_dir)
_logger.info('Finished training model')


# In[11]:


# Save raw estimated graph, ground truth and observational data after training
np.save('{}/true_graph.npy'.format(output_dir), dataset.W)
np.save('{}/X.npy'.format(output_dir), dataset.X)
np.save('{}/final_raw_estimated_graph.npy'.format(output_dir), W_est)


# In[12]:


W_est


# In[13]:


# Plot raw estimated graph
plot_estimated_graph(W_est, dataset.W,
                     save_name='{}/raw_estimated_graph.png'.format(output_dir))

results = count_accuracy(dataset.W, W_est)
_logger.info('Results: {}'.format(results))

_logger.info('Thresholding.')
# Plot thresholded estimated graph
args.graph_thres = 0.3
copy_W_est = np.copy(W_est)
copy_W_est[np.abs(copy_W_est) < args.graph_thres] = 0   # Thresholding
plot_estimated_graph(copy_W_est, dataset.W,
                     save_name='{}/thresholded_estimated_graph.png'.format(output_dir))
results_thresholded = count_accuracy(dataset.W, copy_W_est)
_logger.info('Results after thresholding by {}: {}'.format(args.graph_thres, results_thresholded))

end = time.time()

_logger.info('The time used to execute this is given below')
_logger.info(end - start)

# In[14]:


c = np.sum(copy_W_est, axis=1)
c.shape


# In[15]:


copy_W_est


# In[16]:


#result_matrix = pd.read_csv("W_true.csv", header=None)


# In[17]:


from lingam.utils import make_dot

dot = make_dot(copy_W_est, labels=headers, lower_limit=0.5)

# Save png
dot.format = 'png'
dag_path = dot.render('{}/dag'.format(output_dir))

#dot.render('{}/dag'.format(output_dir))

from IPython.display import Image
Image(filename=dag_path) 

