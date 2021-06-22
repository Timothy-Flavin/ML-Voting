# Nice home grown library for getting 
# your AI's to vote on important business

import time

from matplotlib.pyplot import cla
__default_x_train__ = None
__default_y_train__ = None
__default_x_test__ = None
__default_y_test__ = None
__algos__ = list()
__num_classifications__ = 0

class ML_Algo:
  pre_trained = False
  init_func=None
  predict_func=None
  name="default"
  train_score = 0
  test_score = 0
  model   = None
  x_train = None
  y_train = None
  x_test  = None
  y_test  = None
  train_time = 0

  def __init__(self, init_function, predict_function, model_name, x_train=None, y_train=None, x_test=None, y_test=None):
    self.init_func = init_function
    self.predict_func = init_function
    self.predict_func = predict_function
    self.name = model_name
    self.x_train =x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test

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

def add_algo(ML_Algo):
  __algos__.append(ML_Algo)
  if ML_Algo.x_train is None:
    ML_Algo.x_train = __default_x_train__
  if ML_Algo.y_train is None:
    ML_Algo.y_train = __default_y_train__
  if ML_Algo.x_test is None:
    ML_Algo.x_test = __default_x_test__
  if ML_Algo.y_test is None:
    ML_Algo.y_test = __default_y_test__

def remove_algo(name):
  #remove by name, problems for later
  __algos__.pop()

def train_algos():
  for i in __algos__: 
    if not i.pre_trained:
      tx = i.x_train
      ty = i.y_train
      tex = i.x_test
      tey = i.y_test

      if tx is None:
        tx = __default_x_train__
      if ty is None:
        ty = __default_y_train__
      if tex is None:
        tex = __default_x_test__
      if tey is None:
        tey = __default_x_test__
      
      start = time.time()
      i.train_score, i.test_score, i.model = i.init_func(tx, ty, tex, tey)
      i.train_time = time.time()-start
      print(f"{i.name}, train: {i.train_score}, test: {i.test_score}")

def __vote__(datapoint):
  class_votes = [0]*__num_classifications__
  abs_class_votes = 0
  for i in __algos__:
    weight = (i.test_score-1.0/__num_classifications__)
    j = i.predict_func(i.model, datapoint) # predict should output a number, not a one-hot encoding
    #print(f'{i.name} prediction: {j}, test score: {i.test_score} weight: {weight}')
    class_votes[j] += weight
    abs_class_votes += abs(weight)

  #print(f'class votes: {class_votes}')
  for i in range(__num_classifications__):
    if abs_class_votes!=0:
      class_votes[i]/=abs_class_votes
  #print(f'class votes /= abs: {class_votes}')
  return class_votes

def predict_one(x):
  # change this so that x is a list
  # return max of class votes
  votes = __vote__(x)
  return votes.index(max(votes))

def validate(x, y):
  
  num_right = 0
  for i in range(len(x)):
    #print(f'y actual: {y[i]}')
    if(predict_one(x[i]) == y[i]):
      num_right+=1
      #print("correct")
    #input()
  print(f'num right/len: {num_right*1.0/len(x)}')
  return num_right*1.0/len(x)

def current_algos():
  for i in __algos__:
    print(f"name: {i.name:15}, train accuracy: {i.train_score:8}, test accuracy: {i.test_score:8}, time: {i.train_time:17}s")