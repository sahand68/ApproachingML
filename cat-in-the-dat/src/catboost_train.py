from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd 
from catboost import CatBoostClassifier, Pool
import numpy as np
import itertools
def feature_engineerig(df, cat_cols):

    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:,c1+"_"+c2] = df[c1].astype(str) + "_"+df[c2].astype(str)

    return df


def run(fold):

    df = pd.read_csv('../input/adult_folds.csv')

    num_cols = [
        'fnlwgt','age','capital.gain','capital.loss','hours.per.week'
    ]

    cat_cols = [c for c in df.columns if c not in num_cols and c not in ("kfold","income")]
    #df = df.drop(num_cols, axis = 1)
    df = feature_engineerig(df, cat_cols)
    
    target_mapping = {
        '<=50K' : 0,">50K" : 1
    }

    df.loc[:,"income"] = df.income.map(target_mapping)

    features = [f for f in df.columns if f not in ("kfold", "income")]
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold ==fold].reset_index(drop = True)
    X_train = df_train[features]
    
    X_valid= df_valid[features]
    
    categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

    train_pool =Pool(X_train, df_train.income.values, cat_features = categorical_features_indices)
    valid_pool = Pool(X_valid, df_valid.income.values, cat_features = categorical_features_indices)


    
    model = CatBoostClassifier(cat_features = categorical_features_indices, iterations=1000, 
                           task_type="GPU",
                           devices='0:1', 
                           eval_metric = 'AUC')

    model.fit(train_pool ,use_best_model=True,eval_set = valid_pool)

    valid_preds = model.predict_proba(valid_pool)[:,1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold= {fold}, auc = {auc}")
if __name__=="__main__":
    for fold_ in range(5):
        run(fold_)
