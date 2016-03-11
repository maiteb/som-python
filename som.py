import argparse

import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt

def _parse_file_argument():
    parser = argparse.ArgumentParser("MDS Commandline Tool")
    parser.add_argument('--csv',
                        help='csv filename')
    parser.add_argument('--label_prefix',
                        help='csv\'s label column prefix')
    parser.add_argument('--label_sufix',
                        help='csv\'s label column sufix')
    args = parser.parse_args()
    return args

def _plot_distribution(som):
    fig = plt.figure()
    ax = plt.subplot(aspect='equal')

    plt.pcolor(som.distance_map().T)

    return fig, ax

RS = 20160101

if __name__ == '__main__':
    args = _parse_file_argument()
    data = pd.read_csv(args.csv)
    data.fillna(0, inplace=True)

    label_column = args.label_prefix
    label_prefix = data[label_column].values
    data.drop(label_column, axis=1, inplace=True)

    label_column = args.label_sufix
    label_sufix = data[label_column].values
    data.drop(label_column, axis=1, inplace=True)

    id_column = 'id'
    data.drop(id_column, axis=1, inplace=True)

    som = MiniSom(8,8,len(data.columns),sigma=1.0,learning_rate=0.5,random_seed=RS)
    som.random_weights_init(data.as_matrix())

    som.train_random(data.as_matrix(),100)

    _plot_distribution(som)
    plt.savefig('som.png', dpi=120)
