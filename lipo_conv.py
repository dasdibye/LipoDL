
# This is the code that runs fine on Google Colab tody
# for mol2vec and Lipophilicity

# We have the code for both the MLRegressor MLP n-layers and LSTM

import numpy as np
import pandas as pd
import sys, os
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from mol2vec import features
from mol2vec import helpers

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

from numpy.random import seed
import tensorflow as tf
from tensorflow.keras import layers
seed(123)

def evaluation(model, X_test, y_test):
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)

    plt.figure(figsize=(15, 10))
    fig, ax = plt.subplots()

    ax.scatter(prediction[:400], y_test[:400], c='r', label="prediction_vs_original", linewidth=1.0)
    #plt.scatter(y_test[:400], 'green', label="actual", linewidth=1.0)
    plt.legend()
    #plt.xlabel('Test Set')
    ax.set_xlabel('Actual logP')
    #plt.ylabel('logP')
    ax.set_ylabel('Predicted logP')
    plt.title("MAE {}, MSE {}".format(round(mae, 4), round(mse, 4)))
    plt.show()

    print('MAE score:', round(mae, 4))
    print('MSE score:', round(mse,4))

mdf= pd.read_csv('Lipophilicity_df_revised.csv')
target = mdf['exp']

mdf.drop(columns='exp',inplace=True)
mdf['mol'] = mdf['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
#Loading pre-trained model via word2vec
from gensim.models import word2vec
model = word2vec.Word2Vec.load('model_300dim.pkl')

mols = MolSentence(mol2alt_sentence(mdf['mol'][1], radius=1))
keys = set(model.wv.vocab.keys())
print(keys)
print("*****")
print(set(mols))
print("*****")
mnk = set(mols)&keys
print(mnk)

s2v = sentences2vec(MolSentence(mol2alt_sentence(mdf['mol'][1], radius=1)), model, unseen='UNK')

mdf['sentence'] = mdf.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
mdf['mol2vec'] = [DfVec(x) for x in sentences2vec(mdf['sentence'], model, unseen='UNK')]

X = np.array([x.vec for x in mdf['mol2vec']])
X.shape
y = target.values
y.shape

#For the full training set using the substructure of vectors
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
model = word2vec.Word2Vec.load('model_300dim.pkl')
Xvecall = np.zeros((X.shape[0],101,300))
Xvecall0 = np.zeros((X.shape[0],100,300))
yvecall = np.zeros((y.shape))

print(Xvecall.shape)
print(yvecall.shape)

#train
#for i in range(1,X.shape[0]):
for i in range(X.shape[0]):
  print (i)
  print (mdf['sentence'][i])

  mols = MolSentence(mol2alt_sentence(mdf['mol'][i], radius=1))
  mols0 = MolSentence(mol2alt_sentence(mdf['mol'][i], radius=0))

  xvecapp = np.zeros((100,300))
  xvecapp0 = np.zeros((100,300))

  ii = 0

  # carry the mol2vec on the 0th index of the sequence
  Xvecall[i][0] = X[i]
  for yy in set(mols):
     if yy in keys:
        s2v1 = model.wv.word_vec(yy)
        xvecapp[ii] = s2v1
        ii = ii + 1

  ii = 0
  for yy in set(mols0):
     if yy in keys:
        s2v0 = model.wv.word_vec(yy)
        xvecapp0[ii] = s2v0
        ii = ii + 1

  #for j in range(1,100):
    #for k in range(1,300):
  for j in range(100):
    for k in range(300):
        Xvecall[i][j+1][k] = xvecapp[j][k]

  if ( y[i] < 0 ):
      y[i] = 0.0

  yvecall[i] = y[i]

print('Training tensor shape', Xvecall.shape)
Xvec_train, Xvec_test, yvec_train, yvec_test = train_test_split(Xvecall, yvecall, test_size=.1, random_state=1)




# !!!!!!!!!!!! IMP you need to have the X_val then X_test !!!!!!!!!!!!!!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=1)

n_epoch=40
n_batch=100
seqsize = 100
seq_inputs = layers.Input(shape=(101,300,), dtype='float32')

sin1 = seq_inputs[:,1:101,:]
sin2 = seq_inputs[:,0:1,:]

#
#Layer (type)                    Output Shape         Param #     Connected to
#==================================================================================================
#input_1 (InputLayer)            [(None, 101, 300)]   0
#__________________________________________________________________________________________________
#tf_op_layer_strided_slice (Tens [(None, 100, 300)]   0           input_1[0][0]
#__________________________________________________________________________________________________
#conv1d (Conv1D)                 (None, 99, 1700)     1021700     tf_op_layer_strided_slice[0][0]
#__________________________________________________________________________________________________
#conv1d_1 (Conv1D)               (None, 100, 1700)    511700      tf_op_layer_strided_slice[0][0]
#__________________________________________________________________________________________________
#global_max_pooling1d (GlobalMax (None, 1700)         0           conv1d[0][0]
#__________________________________________________________________________________________________
#global_max_pooling1d_1 (GlobalM (None, 1700)         0           conv1d_1[0][0]
#__________________________________________________________________________________________________
#concatenate (Concatenate)       (None, 3400)         0           global_max_pooling1d[0][0]
#                                                                 global_max_pooling1d_1[0][0]
#__________________________________________________________________________________________________
#dense (Dense)                   (None, 400)          1360400     concatenate[0][0]
#__________________________________________________________________________________________________
#dense_1 (Dense)                 (None, 200)          80200       dense[0][0]
#__________________________________________________________________________________________________
#dense_2 (Dense)                 (None, 1)            201         dense_1[0][0]
#==================================================================================================
#Total params: 2,974,201
#Trainable params: 2,974,201
#Non-trainable params: 0
#__________________________________________________________________________________________________
#None
#(420, 101, 300)
#(420, 1)
#(420,)
#('r2 score = ', 0.6533000333535123)
#('MAE score:', 0.4781)
#('MSE score:', 0.4132)

#conv1 = layers.Conv1D(1700, 2, activation='relu')(sin1)
#pool1 = layers.GlobalMaxPooling1D()(conv1)
#conv2 = layers.Conv1D(1700, 1, activation='relu')(sin1)
#pool2 = layers.GlobalMaxPooling1D()(conv2)
#pool12 = layers.concatenate([pool1, pool2])
#fcoutput1  = (layers.Dense(400,activation="relu"))(pool12)
#fcoutput1  = (layers.Dense(200,activation="relu"))(fcoutput1)
#fcoutput  = (layers.Dense(1,activation="relu"))(fcoutput1)

conv1 = layers.Conv1D(500, 2, activation='relu')(sin1)
pool1 = layers.GlobalMaxPooling1D()(conv1)
conv2 = layers.Conv1D(500, 1, activation='relu')(sin1)
pool2 = layers.GlobalMaxPooling1D()(conv2)
pool12 = layers.concatenate([pool1, pool2])
fcoutput1  = (layers.Dense(200,activation="relu"))(pool12)
fcoutput  = (layers.Dense(1,activation="relu"))(fcoutput1)


#########################################

model = tf.keras.Model(inputs=seq_inputs, outputs=fcoutput)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mean_squared_error'])
history = model.fit(Xvec_train, yvec_train, epochs=n_epoch, batch_size=n_batch, verbose=2,shuffle=True, validation_split=0.11)
#  "Accuracy"
plt.plot(history.history['mean_squared_error'], label='MSE (training data)')
plt.plot(history.history['val_mean_squared_error'], label='MSE (validation data)')
plt.title('MSE values for Lipophilicity Dataset')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


prediction = model.predict(Xvec_test)
print(model.summary())
print(Xvec_test.shape)
print(prediction.shape)
print(yvec_test.shape)
print('r2 score = ', r2_score(yvec_test, prediction))
evaluation(model, Xvec_test,yvec_test)

