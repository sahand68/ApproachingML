from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd 


def run(fold):

    df = pd.read_csv('../input/cat_train_folds.csv')
    features = [f for f in df.columns if f not in ("id", "target","kfold")]
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")
    print(df.kfold.value_counts())
    df_train = df[ df.kfold != fold].reset_index(drop =True)
    df_valid = df[df.kfold == fold].reset_index(drop= True)
    print(df_valid.shape)
    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat([df_train[features], df_valid[features]], axis = 0)

    ohe.fit(full_data[features])

    X_train =ohe.transform(df_train[features])
    X_valid = ohe.transform(df_valid[features])

    model = linear_model.LogisticRegression()

    model.fit(X_train,df_train.target.values)

    valid_preds = model.predict_proba(X_valid)[:,1]

    auc = metrics.roc_auc_score(df_valid.target.values,valid_preds)

    print(f"FOLD = {fold}, AUC= {auc}")

if __name__=="__main__":
    for fold_ in range(5):
        run(fold_)
