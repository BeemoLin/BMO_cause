import sys
import yaml
import argparse


def load_yaml_config(path, skip_lines=0):
    with open(path, 'r') as infile:
        for i in range(skip_lines):
            # Skip some lines (e.g., namespace at the first line)
            _ = infile.readline()

        return yaml.safe_load(infile)


def save_yaml_config(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

class TrainArgs:
    seed = 1230
    n = 1000
    d = 20
    graph_type = 'erdos-renyi'
    degree = 4
    sem_type = 'linear-gauss'
    noise_scale = 1.0
    dataset_type = 'linear'
    l1_lambda = 0.0
    use_float64 = False
    learning_rate = 1e-3
    max_iter = 20
    iter_step = 1500
    init_iter = 3
    h_tol = 1e-8
    init_rho = 1.0
    rho_max = 1e+16
    h_factor = 0.25
    rho_multiply = 10.0
    graph_thres = 0.3

def get_train_args():
    return TrainArgs()

def get_args():
    parser = argparse.ArgumentParser()

    ##### General settings #####
    parser.add_argument('--seed',
                        type=int,
                        default=1230,
                        help='Reproducibility')

    ##### Dataset settings #####
    parser.add_argument('--n',
                        type=int,
                        default=1000,
                        help='Number of observation data')

    parser.add_argument('--d',
                        type=int,
                        default=20,
                        help='Number of nodes')

    parser.add_argument('--graph_type',
                        type=str,
                        default='erdos-renyi',
                        help='Type of graph [erdos-renyi, barabasi-albert]')

    parser.add_argument('--degree',
                        type=int,
                        default=4,
                        help='Degree of graph')

    parser.add_argument('--sem_type',
                        type=str,
                        default='linear-gauss',
                        help='Type of sem [linear-gauss, linear-exp, linear-gumbel ]')

    parser.add_argument('--noise_scale',
                        type=float,
                        default=1.0,
                        help='Variance for Gaussian Noise')

    parser.add_argument('--dataset_type',
                        type=str,
                        default='linear',
                        help='Type of dataset [only linear is implemented]')

    ##### Model settings #####
    parser.add_argument('--l1_lambda',
                        type=float,
                        default=0.0,
                        help='L1 penalty for sparse graph. Set to 0 to disable')

    parser.add_argument('--use_float64',
                        type=bool,
                        default=False,
                        help='Whether to use tf.float64 or tf.float32 during training')

    ##### Training settings #####
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate')

    parser.add_argument('--max_iter',
                        type=int,
                        default=20,
                        help='Number of iterations for optimization problem')

    parser.add_argument('--iter_step',
                        type=int,
                        default=1500,
                        help='Number of steps for each iteration')

    parser.add_argument('--init_iter',
                        type=int,
                        default=3,
                        help='Initial iteration to disallow early stopping')

    parser.add_argument('--h_tol',
                        type=float,
                        default=1e-8,
                        help='Tolerance of optimization problem')

    parser.add_argument('--init_rho',
                        type=float,
                        default=1.0,
                        help='Initial value of rho')

    parser.add_argument('--rho_max',
                        type=float,
                        default=1e+16,
                        help='Maximum value of rho')

    parser.add_argument('--h_factor',
                        type=float,
                        default=0.25,
                        help='Factor of h')

    parser.add_argument('--rho_multiply',
                        type=float,
                        default=10.0,
                        help='Multiplication to amplify rho each time')

    ##### Other settings #####
    parser.add_argument('--graph_thres',
                        type=float,
                        default=0.3,
                        help='Threshold to filter out small values in graph')

    return parser.parse_args(args=sys.argv[1:])
