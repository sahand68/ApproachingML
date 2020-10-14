import os
import numpy as np
from PIL import Image 
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from tqdm import tqdm
import pandas as pd

def create_datasets(trainig_df, image_dir):

    images = []

    targets = []

    for index , row in tqdm(training_df.iterrows(),\
        total = len(training_df), desc = 'processing images'):
        image_id = row['ImageId']
        image_path = os.path.join(image_dir, image_id)

        image = Image.open(image_path + ".png")

        image = image.resize((256, 256), resample = Image.BILINEAR)
        image = np.array(image)

        image= image.ravel()

        images.append(image)
        targets.append(int(row["target"]))

    images = np.array(images)
    print(images.shape)

    return images, targets

if __name__ == "__main__":

    csv_path ='../input/train.csv'
    image_path = '../input/train_png/'

    df= pd.read_csv(csv_path)

    df['kfold '] =-1

    df= df.sample(frac = 1).reset_index(drop = True)

    y = df.target.values

    kf = model_selection.StratifiedKFold(nsplits = 5)

    for f , (t_, v_) in enumerate(kf.split(X= df,y = y)):
        df.loc[v_, 'kfold'] = f

    for fold_ in range(5):

        train_df = df[df.kfold != fold_].reset_index(drop = True)

        test_df = df[df.kfold == fold_].reset_index(drop = True)

        Xtrain,ytrain = create_datasets(train_df, image_path)

        Xtest, ytest= create_datasets(test_df, image_path)

        clf = ensemble.RandomForestClassifier(n_jobs = -1)

        clf.fit(Xtrain,ytrain)

        preds= clf.predict(Xtest)[:,1]

        print(f"Fold: {fold_}")
        print(f"auc = {metrics.roc_auc_score(ytest,preds)}")
        print("")


