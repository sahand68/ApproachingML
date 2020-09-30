import os
import gc
import joblib
import pandas as pd 
import numpy as np
from sklearn import metrics, preprocessing
import fastai
from fastai.tabular.data import *
from fastai.tabular.all import *
from fastai.tabular import *

from pathlib import Path
def run():
    PATH =Path('../input/train.csv')
    dat = pd.read_csv('../input/train.csv', index_col='id')

    cat_names = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 
             'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
            'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']

    cont_names = []
    dep_var = ['target']
    procs = [FillMissing, Categorify, Normalize]
    FillMissing.FillStrategy='MEAN'

    data = TabularDataLoaders.from_df(dat, path=PATH,
                            cat_names=cat_names, 
                            cont_names=cont_names,
                            procs=procs, valid_idx =list(range(len(dat)-50000, len(dat))) , y_names = dep_var, bs=512)
  

    learn = tabular_learner(data, layers=[200,100], metrics=[accuracy], ps=0.15)

    learn.fit_one_cycle(3)

if __name__ == "__main__":
    run()
    