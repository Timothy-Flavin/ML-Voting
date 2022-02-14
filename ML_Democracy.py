# Nice home grown library for getting 
# your AI's to vote on important business

#even the voting algo needs some industrial strength libraries ;)
from argparse import ArgumentError
import time
import numpy as np
import random

from scipy.sparse import data
__default_x_train__ = None
__default_y_train__ = None
__default_x_test__ = None
__default_y_test__ = None
__algos__ = list()
__num_classifications__ = 0

class ML_Algo:
  pre_trained = False
  init_func = None
  predict_func = None
  name="default"
  train_score = 0
  test_score = 0
  model   = None
  x_train = None
  y_train = None
  x_test  = None
  y_test  = None
  train_time = 0
  indices = None
  axis = -1

  def __init__(self, init_function, predict_function, model_name, x_train=None, y_train=None, x_test=None, y_test=None, axis=-1):
    self.init_func = init_function
    self.predict_func = init_function
    self.predict_func = predict_function
    self.name = model_name
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.indices=None
  
  def copy(self):
    cp = ML_Algo(self.init_function, self.predict_function, self.model_name, self.x_train, self.y_train, self.x_test, self.y_test, axis=-1)
    return cp
def set_default_data(x_train, x_test, y_train, y_test):
  global __default_x_train__ 
  global __default_x_test__  
  global __default_y_train__ 
  global __default_y_test__ 
  __default_x_train__ = x_train
  __default_x_test__  = x_test
  __default_y_train__ = y_train
  __default_y_test__  = y_test

def set_num_classifications(num):
  global __num_classifications__ 
  __num_classifications__ = num

def __add_algo__(ML_Algo):
  __algos__.append(ML_Algo)
  if ML_Algo.x_train is None:
    ML_Algo.x_train = __default_x_train__
  if ML_Algo.y_train is None:
    ML_Algo.y_train = __default_y_train__
  if ML_Algo.x_test is None:
    ML_Algo.x_test = __default_x_test__
  if ML_Algo.y_test is None:
    ML_Algo.y_test = __default_y_test__

def add_algo(ML_Algo, num=1):
  if num==1:
    __add_algo__(ML_Algo)
  else:
    for i in range(num):
      al = ML_Algo.copy()
      al.name = al.name + " " + str(i)
      __add_algo__(al)

def remove_algo(name):
  #remove by name, problems for later
  __algos__.pop()

def remove_all_algos():
  global __algos__ 
  __algos__ = list()

def train_algos(output=False, featureProportion=-1.0):
  if featureProportion>1.0:
    raise ArgumentError("Feature proportion cannot be more than 1.0")
  for i in __algos__: 
    if not i.pre_trained:
      tx = i.x_train
      ty = i.y_train
      tex = i.x_test
      tey = i.y_test
      indices=list()
      if(featureProportion>=0):
        indices = list(random.sample(list(np.arange(0,tx.shape[i.axis])),int(tx.shape[i.axis]*featureProportion)))
        indices.sort()
        i.indices=indices
      else:
        indices = None
        i.indices=None
      #print(indices)
      if tx is None:
        tx = np.take(__default_x_train__, indices,axis=i.axis)
      if ty is None:
        ty = __default_y_train__
      if tex is None:
        tex = np.take(__default_x_test__, indices,axis=i.axis)
      if tey is None :
        tey = __default_y_test__
      
      if(featureProportion>=0):
        tx = np.take(__default_x_train__, indices,axis=i.axis)
        tex = np.take(__default_x_test__, indices,axis=i.axis)

      start = time.time()
      i.train_score, i.test_score, i.model = i.init_func(tx, ty, tex, tey)
      i.train_time = time.time()-start
      if output:
        print(f"{i.name}, train: {i.train_score}, test: {i.test_score}, time: {i.train_time}")

def __vote__(datapoint, method):
  class_votes = np.zeros((__num_classifications__, len(datapoint)), dtype='float32')
  abs_class_votes = np.zeros(len(datapoint), dtype='float32')
  for i in __algos__:
    weight = 1.0
    if method==1:
      weight = 1.0/(1.01-i.test_score)#(i.test_score-1.0/__num_classifications__) #makes models that are worse than random have a negative vote
    elif method==2:
      weight = (i.test_score-1.0/__num_classifications__)
    
    if i.indices==None:
      predictions = i.predict_func(i.model, datapoint) # predict should output a number, not a one-hot encoding
    else:
      predictions = i.predict_func(i.model, np.take(datapoint, i.indices,axis=i.axis))
    #print(predictions)
    #print(f'{i.name} prediction: {j}, test score: {i.test_score} weight: {weight}')
    for j in range(len(datapoint)):
      class_votes[predictions[j],j] += weight
      abs_class_votes[j] += abs(weight)
  #print(f'class vote shape: {class_votes.shape}')
  #print(f'class votes: {class_votes[:,0]}')
  for i in range(len(datapoint)):
    if abs_class_votes[i]!=0:
      class_votes[:,i]/=abs_class_votes[i]
      # remember that class_votes[i] is an array of len num_classifications + 1
      # and the last col will be all ones now so we can remove it 
  #class_votes = np.delete(class_votes, __num_classifications__,axis=1)
  #print(f'class votes /= abs: {class_votes[:,0]}')
  #print(f"class votes: {class_votes}")
  if(len(datapoint)==1):
    return class_votes[0]
  return class_votes

def predict_one(x, method):
  # change this so that x is a list
  # return max of class votes
  votes = __vote__([x], method)
  return votes.index(max(votes))

def predict(x, method):
  # return max of class votes for each x
  votes = __vote__(x, method)
  return np.argmax(votes, axis=0)

def predict_no_vote(x):
  predictions = {}
  for i in __algos__:
    predictions[i.name]=i.predict_func(i.model,x=x)
  return predictions

def validate_voting(x, y, method=1):
  print("Validating voting ensemble method...")
  num_right = 0
  predictions = predict(x, method)
  #print(f'shape of x: {x.shape}, len x: {len(x)}')
  for i in range(len(x)):
    #print(f'y actual: {y[i]}')
    if(predictions[i] == y[i]):
      num_right+=1
      #print("correct")
    #input()
  print(f'Score: {num_right*1.0/len(x)}')
  return num_right*1.0/len(x)

def append_model_outputs(x, debug=False):
  if debug:
    print(f"Shape before appending predictions: {x.shape}")
  new_x = np.zeros((len(x), __num_classifications__*len(__algos__)), 'float32')
  for i in range(len(__algos__)):
    predictions = None
    if __algos__[i].indices is None:
      predictions = __algos__[i].predict_func(__algos__[i].model, x)
      
    else: 
      predictions = __algos__[i].predict_func(__algos__[i].model, np.take(x, __algos__[i].indices,axis=i.axis))
    for j in range(len(predictions)):
      new_x[j,predictions[j]+__num_classifications__*i]=1.0
  new_x = np.concatenate((x, new_x), 1)
  if debug:
    print(f"Shape after appending predictions: {new_x.shape}")
  return new_x

def validate_funnel(x_train, y_train, x_test, y_test, classifier, name="funnel model", debug=False):
  if debug:
    print("Validating funnel ensemble method...")
  
  x_train_n = append_model_outputs(x_train)
  x_test_n = append_model_outputs(x_test)
  #print(f'shape of x: {x.shape}, len x: {len(x)}')
  start_t = time.time()
  classifier = classifier.fit(x_train_n, y_train)
  train_score = classifier.score(x_train_n, y_train)
  test_score = classifier.score(x_test_n, y_test)
  print(f"name: {name:15}, train accuracy: {train_score:8}, test accuracy: {test_score:8}, time: {time.time()-start_t:17}s")
  return test_score

def current_algos():
  for i in __algos__:
    print(f"name: {i.name:15}, train accuracy: {i.train_score:.4f}, test accuracy: {i.test_score:.4f}, time: {i.train_time:17}s")

def current_algos_raw():
  algosList = list()
  for i in __algos__:
    algodict={}
    algodict['name']  = i.name
    algodict['train'] = i.train_score
    algodict['test']  = i.test_score
    algodict['time']  = i.train_time
    algosList.append(algodict)
  return algosList