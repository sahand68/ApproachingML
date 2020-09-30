import joblib
import pandas as  pd 
from sklearn import metrics
from sklearn import tree
import config
import argparse
import os
import model_dispacher

def run(fold, model):

    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    X_train =df_train.drop('label', axis = 1).values
    y_train = df_train.label.values

    X_valid = df_valid.drop('label', axis = 1).values
    y_valid = df_valid.label.values

    clf = model_dispacher.models[model]
    clf.fit(X_train,y_train)

    preds = clf.predict(X_valid)
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"fold = {fold}, accuracy={accuracy}")

    joblib.dump(clf,os.path.join(config.MODEL_OUTPUT,f"df_{fold}.bin"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type = int)
    parser.add_argument("--model", type =str)

    args = parser.parse_args()
    run(fold = args.fold, model = args.model)


