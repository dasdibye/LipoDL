# LipoDL
DL models for Lipophilicity prediction.

The repository uploaded consists of various DL models to predict logP Lipophilicity values. It uses the Lipophilicity database provided by DeepChem for training/prediction.
The models use Mol2vec encoding for the input molecules. The files uploaded here consist of two classes - 1) where the training/prediction happens by setting the negative logP 
values to ZERO in order to have a better prediction RMSE and 2) where the full range of logP values are utilized by using Ymin scaling. 

Class 1 consists of a) lipo_MLP_deep.py b) lipo_conv.py c) lipo_lstm.py and d) lipo_ensemble.py
Class 2 consists of a) lipo_MLP_deep_scaled.py b) lipo_conv_scaled.py c) lipo_lstm_scaled.py and d) lipo_ensemble_scaled.py

The lipophilicity database consisting of 4200 molecules is also provided as a .csv file
To run you would also need the file model_300dim.pkl for the 300-dim Mol2vec encoding. 
The link for that file is here: https://github.com/samoturk/mol2vec_notebooks/blob/master/Notebooks/model_300dim.pkl

You can run these Python models on a Linux Ubuntu system which has Tensorflow.
