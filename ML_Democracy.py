# Nice home grown library for getting 
# your AI's to vote on important business

#even the voting algo needs some industrial strength libraries ;)
import time
import numpy as np
import random
import warnings

class ML_Algo:

  def __default_transform__(self, tx,ty):
    return tx, ty

  def __init__(self, fit_function, predict_function, model_name, x_train=None, y_train=None, x_val=None, y_val=None, transform=None, pre_trained=False):
    self.fit_func = fit_function
    self.predict_func = predict_function
    self.name = model_name
    self.indices=None
    if transform is None:
      self.transform = self.__default_transform__
    else:
      self.transform = transform
    self.train_score = 0
    self.val_score = None
    self.test_score = 0
    self.train_time = 0
    self.pre_trained = pre_trained
    self.model = None
    self.fast=False
    self.seed=0
    
  def copy(self):
    cp = ML_Algo(self.fit_func, self.predict_func, self.name, self.transform)
    return cp

class ML_Democracy:
  __algos__ = list()
  __num_classifications__ = 0

  def set_default_data(self, x_train, x_val, y_train, y_val):
    self.__default_x_train__ = x_train
    self.__default_x_val__  = x_val
    self.__default_y_train__ = y_train
    self.__default_y_val__  = y_val

  def set_num_classifications(self, num):
    self.__num_classifications__ = num

  def __add_algo__(self, ML_Algo):
    for i in self.__algos__:
      if i.name == ML_Algo.name:
        raise ValueError(f'Algo to be added "{ML_Algo.name}"\'s name is not unique in the populice')
    self.__algos__.append(ML_Algo)
    
  def add_algo(self, ML_Algo, num=1, verbose=False):
    if num==1:
      self.__add_algo__(ML_Algo)
      if verbose:
        print(f"Added: {ML_Algo.name}")
    else:
      for i in range(num):
        al = ML_Algo.copy()
        al.name = al.name + " " + str(i)
        self.__add_algo__(al)
        if verbose:
          print(f"Added: {al.name}")

  def remove_algo(self, name):
    #remove by name, problems for later
    for i,a in enumerate(self.__algos__):
      if a.name == name:
        self.__algos__.pop(i)
        return

  def get_algo_names(self):
    names = list()
    for i in self.__algos__:
      names.append(i.name)
    return names

  def remove_all_algos(self):
    self.__algos__ = list()

  def train_algos(self, featureProportion=-1.0, bag=False, fast=False, axis=-1, x_train=None, y_train=None, x_val=None, y_val=None, verbose=False):
    if featureProportion>1.0:
      raise ValueError(f"Feature proportion of {featureProportion} cannot be more than 1.0")
    for i in self.__algos__: 
      if not i.pre_trained:
        if x_train is not None:
          tx = x_train
        elif self.__default_x_train__ is not None:
          tx = self.__default_x_train__
        if y_train is not None:
          ty = y_train
        elif self.__default_y_train__ is not None:
          ty = self.__default_y_train__
        if x_val is not None:
          tvx = x_val
        elif self.__default_x_val__ is not None:
          tvx = self.__default_x_val__
        if y_val is not None:
          tvy = y_val
        elif self.__default_y_val__ is not None:
          tvy = self.__default_y_val__

        if tx is None or ty is None:
          raise ValueError(f"Cannot fit model \"{i.name}\" because tx or ty is None. Call \"set_default_data\" or use \"train_algos(x_train=, y_train=)\" arguments")

        indices=list()
        if(featureProportion>=0 and featureProportion<=1.0 and not fast):
          indices = list(random.sample(list(np.arange(0,tx.shape[i.axis])),int(tx.shape[i.axis]*featureProportion)))
          indices.sort()
          i.indices=indices
          i.fast=False
          i.featureProportion = featureProportion
          tx = np.take(tx, indices,axis=axis)
          if tvx is not None:
            tvx = np.take(tvx, indices,axis=axis)
        elif (featureProportion>=0 and featureProportion<=1.0 and fast):
          i.featureProportion = featureProportion
          n = int(tx.shape[axis]*featureProportion)
          seed = random.randint(0,tx.shape[axis]-n)
          tx = np.apply_along_axis(lambda a_1d: a_1d[seed:seed+n], axis, tx)
          if tvx is not None:
            tvx = np.apply_along_axis(lambda a_1d: a_1d[seed:seed+n], axis, tvx)
          i.fast=True
          i.seed=seed
        else:
          indices = None
          i.indices = None
          i.fast = False
          i.featureProportion=-1.0

        if bag:
          choices = random.choices(population=list(np.arange(0,len(tx))),k=len(tx))
          tx = tx[choices]
          ty = ty[choices]
          if tvx is not None:
            choices = random.choices(population=list(np.arange(0,len(tvx))),k=len(tvx))
            tvx = tvx[choices]
            tvy = tvy[choices]

        transtart = time.time()
        tx,ty = i.transform(tx, ty)
        if tvx is not None:
          tvx,tvy = i.transform(tvx, tvy)
        i.trans_time = time.time()-transtart

        start = time.time()
        i.train_score, i.val_score, i.model = i.fit_func(tx, ty, tvx, tvy)
        i.train_time = time.time()-start
        if verbose:
          print(f"{i.name}, train: {i.train_score}, val: {i.val_score}, trans_time: {i.trans_time} train_time: {i.train_time}")

  def __vote__(self, data, method, axis=-1):
    class_votes = np.zeros((self.__num_classifications__, len(data)), dtype='float32')
    abs_class_votes = np.zeros(len(data), dtype='float32')
    data = np.array(data)
    for i in self.__algos__:
      weight = 1.0
      if method==1:
        if i.val_score == None:
          warnings.warn(f"Model: \"{i.name}\" does not have a validation score, defaulting to val_score = 0")
          weight = 1.0/1.01
        else:
          weight = 1.0/(1.01-i.val_score)#(i.val_score-1.0/__num_classifications__) #makes models that are worse than random have a negative vote
      elif method==2:
        if i.val_score == None:
          warnings.warn(f"Model: \"{i.name}\" does not have a validation score, defaulting to weight = 0")
          weight = 0
        else:
          weight = (i.val_score-1.0/self.__num_classifications__)
      
      if i.indices==None and i.fast == False:
        tx,_ = i.transform(data, None)
        predictions = i.predict_func(i.model, tx) # predict should output a number, not a one-hot encoding
      elif i.fast == True:
        if i.featureProportion < 0 or i.featureProportion > 1.0:
          raise ValueError(f"Feature proportion: {i.featureProportion} for model \"{i.name}\" not between 0 and 1")
        n = int(data.shape[axis]*i.featureProportion)
        tx = np.apply_along_axis(lambda a_1d: a_1d[i.seed:i.seed+n], axis, data)
        tx,_ = i.transform(tx, None)
        predictions = i.predict_func(i.model, tx)
      else:
        tx = np.take(data, i.indices,axis=axis)
        tx,_ = i.transform(tx, None)
        predictions = i.predict_func(i.model, tx)
      #print(predictions)
      #print(f'{i.name} prediction: {j}, test score: {i.val_score} weight: {weight}')
      for j in range(len(data)):
        class_votes[predictions[j],j] += weight
        abs_class_votes[j] += abs(weight)
    #print(f'class vote shape: {class_votes.shape}')
    #print(f'class votes: {class_votes[:,0]}')
    for i in range(len(data)):
      if abs_class_votes[i]!=0:
        class_votes[:,i]/=abs_class_votes[i]
        # remember that class_votes[i] is an array of len num_classifications + 1
        # and the last col will be all ones now so we can remove it 
    #class_votes = np.delete(class_votes, __num_classifications__,axis=1)
    #print(f'class votes /= abs: {class_votes[:,0]}')
    #print(f"class votes: {class_votes}")
    if(len(data)==1):
      return class_votes[0]
    return class_votes

  def predict_one(self, x, method):
    # change this so that x is a list
    # return max of class votes
    votes = self.__vote__(np.array([x]), method)
    return votes.index(max(votes))

  def predict(self, x, method):
    # return max of class votes for each x
    votes = self.__vote__(x, method)
    return np.argmax(votes, axis=0)

  def predict_no_vote(self, x):
    predictions = {}
    for i in self.__algos__:
      predictions[i.name]=i.predict_func(i.model,x=x)
    return predictions

  def test_models(self, x, y, verbose=False):
    scores = {}
    for i in self.__algos__:
      scores[i.name]=0
      i.test_score=0
      preds = i.predict_func(x)
      for j in range(x.shape[0]):
        if preds[j]==y[j]:
          i.test_score+=1
      i.test_score = i.test_score / x.shape[0]
      scores[i.name]=i.test_score
    if verbose:
      print(scores)
    return scores

  def validate_voting(self, x, y, method=1):
    print("Validating voting ensemble method...")
    num_right = 0
    predictions = self.predict(x, method)
    #print(f'shape of x: {x.shape}, len x: {len(x)}')
    for i in range(len(x)):
      #print(f'y actual: {y[i]}')
      if(predictions[i] == y[i]):
        num_right+=1
        #print("correct")
      #input()
    print(f'Score: {num_right*1.0/len(x)}')
    return num_right*1.0/len(x)

  def append_model_outputs(self, x, debug=False):
    raise ValueError("Not implemented yet")
    if debug:
      print(f"Shape before appending predictions: {x.shape}")
    new_x = np.zeros((len(x), self.__num_classifications__*len(__algos__)), 'float32')
    for i in range(len(__algos__)):
      predictions = None
      if __algos__[i].indices is None:
        predictions = __algos__[i].predict_func(__algos__[i].model, x)
        
      else: 
        predictions = __algos__[i].predict_func(__algos__[i].model, np.take(x, __algos__[i].indices,axis=__algos__[i].axis))
      for j in range(len(predictions)):
        new_x[j,predictions[j]+__num_classifications__*i]=1.0
    new_x = np.concatenate((x, new_x), 1)
    if debug:
      print(f"Shape after appending predictions: {new_x.shape}")
    return new_x

  def validate_funnel(self, x_train, y_train, x_val, y_val, classifier, name="funnel model", debug=False):
    raise ValueError("Not implemented yet")
    if debug:
      print("Validating funnel ensemble method...")
    
    x_train_n = append_model_outputs(x_train)
    x_val_n = append_model_outputs(x_val)
    #print(f'shape of x: {x.shape}, len x: {len(x)}')
    start_t = time.time()
    classifier = classifier.fit(x_train_n, y_train)
    train_score = classifier.score(x_train_n, y_train)
    val_score = classifier.score(x_val_n, y_val)
    print(f"name: {name:15}, train accuracy: {train_score:8}, test accuracy: {val_score:8}, time: {time.time()-start_t:17}s")
    return val_score

  def current_algos(self):
    for i in self.__algos__:
      print(f"name: {i.name:15}, train accuracy: {i.train_score:.4f}, val accuracy: {i.val_score:.4f}, test accuracy: {i.test_score:.4f}, time: {i.train_time:17}s, trans_time: {i.trans_time:18}s")

  def current_algos_raw(self):
    algosList = list()
    for i in self.__algos__:
      algodict={}
      algodict['name']  = i.name
      algodict['train'] = i.train_score
      algodict['val']  = i.val_score
      algodict['test']  = i.test_score
      algodict['time']  = i.train_time
      algodict['trans_time'] = i.trans_time
      algosList.append(algodict)
    return algosList