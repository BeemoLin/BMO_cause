#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# exp_name = 'n_500_d_5_e_3_ER'

# exp_name = 'n_500_d_20_e_12_ER'
# exp_name = 'n_500_d_20_e_12_SF'
# exp_name = 'n_500_d_100_e_60_ER'
# exp_name = 'n_500_d_100_e_60_SF'
# exp_name = 'n_500_d_300_e_180_ER'
# exp_name = 'n_500_d_300_e_180_SF'

exp_name = 'n_2000_d_20_e_12_ER'
# exp_name = 'n_2000_d_20_e_12_SF'
# exp_name = 'n_2000_d_100_e_60_ER'
# exp_name = 'n_2000_d_100_e_60_SF'
# exp_name = 'n_2000_d_300_e_180_ER'
# exp_name = 'n_2000_d_300_e_180_SF'


dataset_path = '../../experiment/datasets/{}/X.csv'.format(exp_name)
true_path = '../../experiment/datasets/{}/W_true.csv'.format(exp_name)

df = pd.read_csv(dataset_path, header=None)
X = pd.DataFrame(df)

model = lingam.DirectLiNGAM()
model.fit(X)

W_est = model.adjacency_matrix_

W_true = pd.read_csv(true_path, header=None)


# In[ ]:


# Setup for logging
output_dir = 'output/{}/{}'.format(exp_name, datetime.now(timezone('Asia/Taipei')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
create_dir(output_dir)
LogHelper.setup(log_path='{}/training.log'.format(output_dir), level_str='INFO')
_logger = logging.getLogger(__name__)


# In[ ]:


# Save w_est png
dot = make_dot(W_est)

dot.format = 'png'

w_est_dag = dot.render('{}/w_est_dag'.format(output_dir))

# Save w_true png
dot = make_dot(W_true)

dot.format = 'png'

w_true_dag = dot.render('{}/w_true_dag'.format(output_dir))


# In[ ]:


model.adjacency_matrix_


# In[ ]:


X


# In[ ]:


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

# In[ ]:


# draw est dag
from IPython.display import Image
Image(w_est_dag) 


# In[ ]:


# draw true dag
from IPython.display import Image
Image(w_true_dag) 

