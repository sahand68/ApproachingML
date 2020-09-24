import pandas as pd  
from sklearn import model_selection
if __name__ =="main":

    df = pd.read_csv('train.csv')
    #we create a new colum called kfoild and fill with -1
    df['kfold'] = -1
    #randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop =True)
    #fetch targets
    y = df.target.values
    #intiate kflold class from model_selction module
    kf = model_selection.StratifiedKFold(n_splits = 5)
    #fill the new kfold column
    for fold,(trn_,val_) in enumerate(kf.split(X=df,y = y)):
        df.loc[val_,'kfold']  = fold
    #save the new csv with kfold column
    df.to_csv('train_folds.csv', index = False)