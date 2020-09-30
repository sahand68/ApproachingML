from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd 
import xgboost as xgb


def run(fold):

    df = pd.read_csv('../input/cat_train_folds.csv')
    features = [f for f in df.columns if f not in ("id", "target","kfold")]
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")
    
    for col in features:
        lb = preprocessing.LabelEncoder()
        lb.fit(df[col])
        df.loc[:,col] = lb.transform(df[col])

    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold ==fold].reset_index(drop = True)
    X_train = df_train[features].values
    X_valid= df_valid[features].values
    model = xgb.XGBClassifier(
        
        n_jobs = 1,
        max_depth = 7,
        n_estimators = 200)        
    model.fit(X_train, df_train.target.values)

    valid_preds = model.predict_porba(X_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"Fold= {fold}, auc = {auc}")
if __name__=="__main__":
    for fold_ in range(5):
        run(fold_)
