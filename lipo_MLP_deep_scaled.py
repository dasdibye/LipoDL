
#
# Lipophilicity prediction using MLP

#Copyright 2021-2022 Dibyendu Das (Intel), Riya Datta (Chris University), Srinjoy Das
#Contact Dibyendu Das<dibyendu.das0708@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
#from keras_self_attention import SeqSelfAttention

#import io

from mol2vec import features
from mol2vec import helpers

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras import layers


def evaluation(model, X_test, y_test_scaled, Ymin):
    prediction_scaled = model.predict(X_test)
    prediction = prediction_scaled + Ymin
    y_test = y_test_scaled + Ymin

    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)

    plt.figure(figsize=(15, 10))
    fig, ax = plt.subplots()

    ax.scatter(prediction[:400], y_test[:400], c='r', linewidth=1.0)
    plt.legend()
    ax.set_xlabel('Real logP')
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
mnk = set(mols)&keys

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

Ymin = 0.0
for i in range(X.shape[0]):
  if ( y[i] < 0 ):
      if ( y[i] < Ymin ) :
          Ymin = y[i]

print('Training tensor shape', Xvecall.shape)
Xvec_train, Xvec_test, yvec_train, yvec_test = train_test_split(Xvecall, yvecall, test_size=.1, random_state=1)

y_scaled = y - Ymin

X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=.2, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=1)

seed(100)

MLPregr = MLPRegressor(hidden_layer_sizes=(300,500,500,600,500,500,100,100,75,50),alpha=0.00005, batch_size=200, random_state=1, max_iter=300)
MLPregr.fit(X_train, y_train)
pd.DataFrame(MLPregr.loss_curve_).plot()

prediction = MLPregr.predict(X_test)
print(X_test.shape)
print(prediction.shape)
print(y_test.shape)
print('r2 score = ', r2_score(y_test, prediction))
evaluation(MLPregr, X_test, y_test, Ymin)
