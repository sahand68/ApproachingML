import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import model_selction

def create_folds(data):

    #we create new columns called kfold and we fill it with -1

    data['kfold'] = -1
    #the next step is to randomized rows of data

    data = data.sample(frac = 1).reset_index(drop = True)
    # calculate number of bins by Sturge's rule

    num_bins = int(np.floor(1+np.log2((len(data)))))
    #bins cut
    data.loc[:,'bins'] = pd.cut(data['target'], bins = num_bins, labels = False)

    #initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits = 5)

    # fill the new columns
    # note that, instead of targets we use bins! 
     
    for f, (t_, v_) in enumerate(kf.split(x= data, y = data.bins.values)):
        data.loc[v_,'kfold'] = f

    #drop the bins column
    data = data.drop('bins', axis = -1)

    return data


if __name__ = "__main__":

    X, y = datasets.make_regression(n_samples = 15000, n_features = 100, n_targets = 1)

    df = pd.DataFrame(x, columns=[f"f_{i}" for i in range(X.shape[-1])])

    df.loc[:,"target"] = y

    df = create_folds(df)


    