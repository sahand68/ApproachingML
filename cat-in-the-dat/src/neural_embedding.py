import os
import gc
import joblib
import pandas as pd 
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as k
from tensorflow.keras import utils

def create_model(data, catcols):

    '''
    this funciton returns a compiled tf.keras model for entitiy embeddings
    :param data: this is a pandas dataframe
    :param catcols: list of categorical column names
    :return: complied tf.keras model
    '''
    #init the list of inputs for embedding
    inputs =[]
    #init the list of outputs for embedding
    outputs= []
    #loop over all categorical columns
    for c in catcols:

        #find the number of unique values in the column
        num_unique_values= int(data[c].nunique())
        #simple dimension of embedding calculator
        #min size is half the number of unbique values
        #max size is 50. max size depends on the number of values
        #categories too. 50 is quite sufficient most of the times
        #but if you have millions of unique values, you might need a larger dimenion

        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))
        #simple keras input layer with size 1

        inp = layers.Input(shape = (1,))
        #add embedding layer to raw input
        #embedding size is alwasy 1 more than unique values in input

        out = layers.Embedding(num_unique_values + 1, embed_dim, name = c)(inp)

        #1-d spatial dropout is the standard for embedding layers
        #it can be used in nlp tasks as well

        out = layers.SpatialDropout1D(0.3)(out)

        #reshape the input to the dimensions of embedding
        #this becomes our output layer for current feature

        out = layers.Reshape(target_shape = (embed_dim,))(out)

        #add input to input list
        inputs.append(inp)
        #add output to output list
        outputs.append(out)


    #concatenate all output layers
    X = layers.Concatenate()(outputs)
    # add a batchnorm layer
    # from here, everything is up to you
    # you can try different architecture
    # add numerical features here or in concatonate layer
    X = layers.BatchNormalization()(X)

    # a bunch of dense layers with dropout
    # start with 1 or two layers only
    X = layers.Dense(300,activation = 'relu')(X)
    X = layers.Dropout(0.3)(X)
    X = layers.BatchNormalization()(X)

    #using softmax and treating it as a two class problem
    # sigmoid can also be used but then we need only 1 output class
    y = layers.Dense(2, activation = 'softmax')(X)

    model = Model(inputs = inputs ,outputs = y)
    #compile the model
    # we use adam and binary cross entropy
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

    return model 
def run(fold):
    df = pd.read_csv('../input/cat_train_folds.csv')
    features = [
        f for f in df.columns if f not in ("id","target","kfold")
    ]
    #fill all Na with NONE
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")
        #encode all features with label encoder individually
        #in a live setting all label encoders need to be saved
        
        for feat in features:
            df.loc[:,feat] = df[feat].astype(str)
            lbl_enc = preprocessing.LabelEncoder()
            lbl_enc = lbl_enc.fit(df[feat].values)
            df.loc[:, feat] = lbl_enc.fit_transform(df[feat].astype(str).values)

        #get trainign data using folds

        df_train= df[df.kfold != fold].reset_index(drop = True)
        df_valid = df[df.kfold ==fold].reset_index(drop = True)

        model = create_model(df, features)
        #our features are a list of list
        Xtrain = [df_train[features].values[:,k] for k in range(len(features))]
        Xvalid = [df_valid[features].values[:,k] for k in range(len(features))]

        ytrain = df_train.target.values
        yvalid = df_train.target.values

        #concert target columns to categories
        #this is just binarization

        ytrain_cat = utils.to_categorical(ytrain)
        yvalid_cat = utils.to_categorical(yvalid)

        #fit the model

        model.fit(Xtrain,ytrain_cat, validation_data = (Xvalid, yvalid_cat), verbose = 1, batch_size =1024, epochs = 3)


        valid_preds = model.predict(Xvalid)[:,1]
        print(metrics.roc_auc_score(yvalid, valid_preds))

        #clear session to free gpu memory

        k.clear_session()

if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)
