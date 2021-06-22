# Grass fed libraries
import ML_Democracy as MLD
# Factory farmed libraries ;)
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf         
from tensorflow import keras
# Spicy models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as sk_logistic
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
# Spicier preprocessing techniques 
from sklearn.decomposition import PCA
# Super Secret Constants for test purposes ;) 
TRAIN_LEN = 10000
TEST_LEN  = 2000

# Data to test with
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(f'original shape upon import, x: {x_train.shape}, y: {y_train.shape}')

x_train = np.reshape(x_train, (60000, 28*28))[:TRAIN_LEN]
x_test = np.reshape(x_test, (10000, 28*28))[:TEST_LEN]
y_train = y_train[:TRAIN_LEN]
y_test = y_test[:TEST_LEN]

print(f'new shape after flattening, x: {x_train.shape}, y: {y_train.shape}')

# Preprocessing
def principal_squish_squash(data):
  tran = PCA(n_components=0.950)
  data_out = tran.fit_transform(data)
  return tran, data_out

# Comment this out to not use PCA
tran, x_train = principal_squish_squash(x_train)
x_test = tran.transform(x_test)
# End of PCA Code

print(f'new shape after preprocessing, x: {x_train.shape}, y: {y_train.shape}')

def mlp_init(x_train, y_train, x_test, y_test):
  mlp = MLPClassifier(hidden_layer_sizes=(256,128,32,), alpha=0.01, max_iter=20).fit(x_train, y_train)
  test_score = mlp.score(x_test, y_test)
  train_score = mlp.score(x_train, y_train)

  return train_score, test_score, mlp

def lda_init(x_train, y_train, x_test, y_test):
  lda = LinearDiscriminantAnalysis().fit(x_train, y_train)
  test_score = lda.score(x_test, y_test)
  train_score = lda.score(x_train, y_train)

  return train_score, test_score, lda 

def nb_init(x_train, y_train, x_test, y_test):
  nb = GaussianNB().fit(x_train, y_train)
  test_score = nb.score(x_test, y_test)
  train_score = nb.score(x_train, y_train)

  return train_score, test_score, nb 

def log_init(x_train, y_train, x_test, y_test):
  log = sk_logistic(max_iter=300).fit(x_train, y_train)
  test_score = log.score(x_test, y_test)
  train_score = log.score(x_train, y_train)

  return train_score, test_score, log 
  
def svm_init(x_train, y_train, x_test, y_test):
  svmp = svm.SVC(tol=0.05).fit(x_train, y_train)
  test_score = svmp.score(x_test, y_test)
  train_score = svmp.score(x_train, y_train)

  return train_score, test_score, svmp 

def rf_init(x_train, y_train, x_test, y_test):
  rf = RandomForestClassifier(n_estimators=500).fit(x_train, y_train)
  test_score = rf.score(x_test, y_test)
  train_score = rf.score(x_train, y_train)

  return train_score, test_score, rf 

def gbc_init(x_train, y_train, x_test, y_test):
  gbc = GradientBoostingClassifier(
    max_depth=3,
    n_estimators=15,
    learning_rate=0.1
  ).fit(x_train, y_train)
  test_score = gbc.score(x_test, y_test)
  train_score = gbc.score(x_train, y_train)

  return train_score, test_score, gbc 

# Will be different for tf models and any other library
def predict_one(model, x):
  return model.predict([x])[0]

MLD.set_num_classifications(10)
MLD.set_default_data(x_train, x_test, y_train, y_test)
MLD.add_algo(MLD.ML_Algo(log_init, predict_one, "logistic reg"))
MLD.add_algo(MLD.ML_Algo(mlp_init, predict_one, "mlp classifier"))
MLD.add_algo(MLD.ML_Algo(lda_init, predict_one, "lda"))
MLD.add_algo(MLD.ML_Algo(nb_init, predict_one, "Nbayes"))
#MLD.add_algo(MLD.ML_Algo(svm_init, predict_one, "support vector"))
MLD.add_algo(MLD.ML_Algo(rf_init, predict_one, "random forest"))
MLD.add_algo(MLD.ML_Algo(gbc_init, predict_one, "gradient boost"))
MLD.train_algos()
MLD.current_algos()
MLD.validate(x_test, y_test)