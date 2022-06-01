# Nice home grown library for getting 
# your AI's to vote on important business

#even the voting algo needs some industrial strength libraries ;)
import time
import numpy as np
import random
import warnings

class ML_Algo:
  """ ML_Algo is a machine learning model/algorithm that is formatted 
  properly for use in an ensamble through ML_Democracy

    Member Methods:

    fit_function: A function taking 4 arguments, training data, training 
    labels, validation data, and validation labels. This function needs 
    to be set as an argument in the ML_Algo's initialization. fit_function
    should return three objects, a training score between 0.0 and 1.0, a
    validation score between 0.0 and 1.0, and a model object. The model
    object can be of any form as long as the predict_function passed to 
    this ML_Algo can use the model and data alone to make predictions. 
    The fit_function should obtain it's scores by training the a model on
    the training data and labels. 

    predict_function: A function taking two arguments, the model returned
    from fit_funtion, and the data on which to make predictions. It should
    return a 1D numpy array of integer class predictions in the same order
    as the data instances provided. 

    transform: ML_Democracy does not assume that models will all expect
    data in the same format, yet ML_Democracy needs to be able to make
    online predictions. The solution is a transform function which each 
    unique ML_Algo should have in order to transform incomming data into
    the particular model's requirements. If no transform is given, the 
    model will not transform the data before training or prediction. 

    copy: Returns another instance of this ML_Algo with the same init
    arguments as this one. 

    Member Variables:

    model_name (string): A unique identifier for this model. This name can be used
    to add and remove or retrain particular models in the ensamble. 

    pre_trained (boolean): If true, this ML_Algo will not have it's 
    fit_function called during ML_Democracy's train_algos method. This 
    is for models which have been pre_trained. 

  """
  def __default_transform__(self, tx,ty):
    """If no transform function is passed to the model, the default is
    used. This transform returns the data it is given."""
    return tx, ty

  def __init__(self, fit_function, predict_function, model_name, transform=None, pre_trained=False):
    """Creates an ML_Algo to be used as a part of the ensamble. See 
    'help(ML_Algo)' for more details"""
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
    """ copy: Returns another instance of this ML_Algo with the same init
    arguments as this one. """
    cp = ML_Algo(self.fit_func, self.predict_func, self.name, self.transform)
    return cp

class ML_Democracy:

  def __init__(self):
    self.__algos__ = list()
    self.__num_classifications__ = 0
    self.featureProportion=None 
    self.bag=None
    self.fast=None
    self.axis=None
  def set_default_data(self, x_train, x_val, y_train, y_val):
    """Sets default dataset to be used for ensamble algorithms. Data
    can be provided here once after initialization of ML_Democracy or 
    data can be passed in with each algorithm. default data is used 
    for any model which is not given individual data.
    
    Arguments:
    x_train: training data. Should be numpy array but can be any
    dimensions
    y_train: training labels, should be 1 dimensional numpy array 
    with integer classes
    x_val: validation data, same format as x_train
    y_val: validation labels, same format as y_train"""
    self.__default_x_train__ = x_train
    self.__default_x_val__  = x_val
    self.__default_y_train__ = y_train
    self.__default_y_val__  = y_val

  def set_num_classifications(self, num):
    """Number of unique classes. Needed for the 'distance from 
    randomness' voting algorithm."""
    self.__num_classifications__ = num

  def __add_algo__(self, ML_Algo):
    for i in self.__algos__:
      if i.name == ML_Algo.name:
        raise ValueError(f'Algo to be added "{ML_Algo.name}"\'s name is not unique in the populice')
    self.__algos__.append(ML_Algo)
    
  def add_algo(self, ML_Algo, num=1, verbose=False):
    """Adds a numbe of algorithms of a given type to the Ensamble. 
    
    Arguments:
    ML_Algo: Algorithm to be added
    num: Number of said algorithm to be added. Name will be given
    an increment number to make algorithms maintain unique names
    verbose: False by default. Set to true for debug informational
    output."""
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
    """Removes an algorithm by name. String must match
    exactly."""
    #remove by name, problems for later

    for i,a in enumerate(self.__algos__):
      if a.name == name:
        self.__algos__.pop(i)
        return

  def get_algo_names(self, verbose=False):
    """Returns a list of algorithm name strings"""
    names = list()
    for i in self.__algos__:
      names.append(i.name)
      if verbose:
        print(i.name)
    return names

  def remove_all_algos(self):
    """Resets ensamble algo list to be empty"""
    self.__algos__ = list()

  def mark_for_retrain(self,name):
    """Given a name to retrain, this will set that algorithm's
    'pretrained' flag to False. The algorithm will now be 
    trained when calling train_algos by calling it's fit_func
    on the training data once again. If bootstrapping is used,
    a new bootstrap sample will be drawn. A new feature subset
    will also be selected."""
    for i in self.__algos__:
      if i.name == name:
        i.pre_trained = False

  def __data_or_default(self,x_train=None, y_train=None, x_val=None, y_val=None):
    tx=None
    ty=None
    tvx=None
    tvy=None
    
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

    return tx,ty,tvx,tvy

  def train_algos(self, featureProportion=-1.0, bag=False, fast=False, axis=-1, x_train=None, y_train=None, x_val=None, y_val=None, verbose=False):
    """Trains all algorithms in the ensamble with their 'pretrained'
    flag set to False. The pretrained flag for each algorithm will
    subsiquently be set to true and validation scores will be 
    recorded if applicable for future use with voting methods. 
    
    Arguments:
    
    feature proportion: [0.0-1.0] will result in a subset of the
    features being used per model in the ensamble. the number of
    features selected is num_features * featureProportion. 
    selection method depends on 'fast' argument value. If feature 
    proportion is set to <0 (-1 by default) all features will 
    be used for each model. 
    
    fast: Boolean. If fast = False, then features will be
    selected randomly by sampling the supplied 'axis' of the input
    array without replacement. If fast = True, then a random 
    contiguous interval along the given axis will be supplied
    instead for the purposes of selecting a contiguous portion
    of an image for example. 

    bag: If true, bootstrap sampling will be used for each model.
    if false, all data will be used on each model. 

    x_train, y_train, x_val, y_val: See 'set_default_data' for
    more information. These arguments are used if there is no 
    default data. 

    verbose: False by default, set to True for debug output. 
    """
    # Add check to make sure args are the same or models are all untrained
    if featureProportion>1.0:
      raise ValueError(f"Feature proportion of {featureProportion} cannot be more than 1.0")
    for i in self.__algos__: 
      if not i.pre_trained:
        tx,ty,tvx,tvy = self.__data_or_default(x_train,y_train,x_val,y_val)

        if tx is None or ty is None:
          raise ValueError(f"Cannot fit model \"{i.name}\" because x_train or y_train is None. Call \"set_default_data\" or use \"train_algos(x_train=, y_train=)\" arguments")

        indices=list()
        if(featureProportion>=0 and featureProportion<=1.0 and not fast):
          indices = list(random.sample(list(np.arange(0,tx.shape[axis])),int(tx.shape[axis]*featureProportion)))
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
        i.pre_trained = True
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
    """Used to predict a single data instance. x shoud be a 
    numpy array of the same shape as a single element of the
    given data and method can be 0 1 or 2 to determine the
    voting method used. See 'help(predict)' for more details
    on method. """
    # change this so that x is a list
    # return max of class votes
    votes = self.__vote__(np.array([x]), method)
    return votes.index(max(votes))

  def predict(self, x, method):
    """Used to predict n array data instance. x shoud be a 
    numpy array of the same shape as the training data 
    and method can be 0 1 or 2 to determine the
    voting method used. 
    
    1. Majority voting: most models win. uppon tie, first 
    class wins
    2. Distance from perfection: Models are weighted based
    on their validation accuracy compared to perfect 
    accuracy. w = 1.01 / (1.0-val_accuracy)
    3. Distance from random: Models are weighted based on 
    their distance from a random classification. 
    w = val_accuracy - (1.0/num_classes)"""
    # return max of class votes for each x
    votes = self.__vote__(x, method)
    return np.argmax(votes, axis=0)

  def predict_no_vote(self, x):
    """Gets predictions on x for each individual model and 
    returns a dictionary of the form {model_name: preds}
    where pres are the predictions made by that model.
    see "predict" for more information."""
    predictions = {}
    for i in self.__algos__:
      predictions[i.name]=i.predict_func(i.model,x=x)
    return predictions

  def test_models(self, x, y, verbose=False):
    """Given a new dataset, test_models will return the 
    accuracy scores of each model on that dataset in a
    dictionary of the form {model_name: score}
    
    Arguments: x, y, verbose
    
    x: Data, of the same form as the data used in 
    training. see 'set_default_data for more information
    
    y: intger class labels for x
    
    verbose: False by default, if set to True, then 
    debug output will be printed."""
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
    """Given new data, returns the accuracy achieved by
    a given voting method, default '1'"""
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
    """Prints sumarry of current algorithms including
    recorded accuracies and training times. """
    for i in self.__algos__:
      print(f"name: {i.name:15}, train accuracy: {i.train_score:.4f}, val accuracy: {i.val_score:.4f}, test accuracy: {i.test_score:.4f}, time: {i.train_time:17}s, trans_time: {i.trans_time:18}s")

  def current_algos_raw(self):
    """Returns sumarry of current algorithms including
    recorded accuracies and training times. Summary returned
    is a list of dictionaries where each dictionary can be 
    compared to a column in a data table with several 
    recorded statistics of interest. meant to be used with 
    pandas data tables for organizing results. """
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

  def auto_retrain(self, std_dist=None, retrain = True, max_iter = 10, featureProportion=-1.0, bag=False, fast=False, axis=-1, x_train=None, y_train=None, x_val=None, y_val=None, verbose=False):
    """Refining method for list of algorithms. If certain 
    algorithms are bad enough, their 'pretrained' flag will
    be set to False and they will be retrained a number of
    times until they meet the required metrics. 
    
    New Arguments: retrain, max_iter, std_dist
    Repeated Arguments from train_algos: featureProportion, 
    bag, fast, axis, x_train, y_train, x_val, y_val, verbose
    
    retrain: True by default. If true, algorithms will be 
    retrained until max_iter is reached. If False, algorithms
    will only be retrained once. 

    max_iter: Number of times algorithms will be analyzed and
    retrained. 

    std_dist: Number of standard deviations below average that
    an algorithm accuracy has to be in order for it to be 
    retrained. This number is not updated each iteration. 
    """
    
    val_scores = np.zeros((len(self.__algos__)))
    for i in range(len(self.__algos__)):
      val_scores[i] = self.__algos__[i].val_score
    mn = np.mean(val_scores)
    st = np.std(val_scores)
    min_acc = mn-st*std_dist

    if verbose:
      print(f"Mean accuracy: {mn}, Standard deviation: {st}, min_acc: {min_acc}")

    trained = False
    tnum = 0

    while not trained:
      trained = True
      for i in range(len(self.__algos__)):
        if verbose:
          print("Algos needing to be trained again: ")
        if self.__algos__[i].val_score < min_acc:
          trained = False
          self.__algos__[i].pre_trained = False
          if verbose:
            print(f"Algo: {self.__algos__[i].name}")
      if not trained and retrain:
        if tnum == max_iter:
          print(f"Maximum iterations, {max_iter}, reached in retrain routine")
          return False
        tnum +=1
        if verbose:
          print(f"re_training iteration {tnum}")
        if x_val is None and self.__default_x_val__ is None:
          raise ValueError("In order to auto-retrain models a validation set is needed, but x_val and default x_val are both None")
        self.train_algos(featureProportion=featureProportion, bag=bag, fast=fast,axis=axis, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        if verbose:
          self.current_algos()
      elif not retrain:
        trained = True
    
    #let the user know if some need to be trained
    trained = True
    for i in range(len(self.__algos__)):
      if self.__algos__[i].val_score < min_acc:
        trained = False
    if verbose:
      print(f"Algorithms converged: {trained}")
    return False

  def auto_prune(self, std_dist=None, verbose=False):
    """Pruning method for list of algorithms. If certain 
    algorithms are bad enough, they will be removed from 
    the ensamble. 
    
    Arguments: std_dist, verbose
    
    std_dist: Number of standard deviations below average that
    an algorithm accuracy has to be in order for it to be 
    retrained.

    verbose: False by default. if True, will print debug info. 
    """
    val_scores = np.zeros((len(self.__algos__)))
    for i in range(len(self.__algos__)):
      val_scores[i] = self.__algos__[i].val_score
    mn = np.mean(val_scores)
    st = np.std(val_scores)
    min_acc = mn-st*std_dist

    if verbose:
      print(f"Mean accuracy: {mn}, Standard deviation: {st}, min_acc: {min_acc}")

    names = []
    for i in range(len(self.__algos__)):
      if verbose:
        print("Algos being pruned: ")
      if self.__algos__[i].val_score < min_acc:
        names.append(self.__algos__[i].name)
        if verbose:
          print(f"Algo: {self.__algos__[i].name}")

  def expensive_prune(self, x_val=None, y_val=None, min_algos=0.5, method=0, verbose=False):
    """Pruning method for list of algorithms. If certain 
    algorithms reduce the ensamble's validation accuracy,
    they will be removed from the ensamble. This method 
    is more expensive because the ensamble will need to 
    be validated n times per removal where n is the 
    number of models in the ensamble. 
    
    Arguments: x_val, y_val, min_algos, method, verbose
    
    x_val: Validation data if default data is not set. 
    None by default. See set_default_data.
    
    y_val: Validation labels if default labels are not set. 
    None by default. See set_default_data.
    
    min_algos: Minimum proportion of algorithms to keep in the 
    ensamble. between 0.0 and 1.0 
    
    method: Voting method to be used.
    
    verbose: False by default. If true, debug output will be
    printed. 
    
    """
    _x,_y,x_val,y_val = self.__data_or_default(None,None,x_val=x_val,y_val=y_val)
    if x_val is None or y_val is None:
      raise ValueError("validation data is required for expensive_prune, either pass x_val and y_val args, or use \"set_default_data\"")

    acc = self.validate_voting(x_val, y_val, method)
    n_algos = len(self.__algos__)
    bacc = acc+1
    algo_to_pop = 0

    while (len(self.__algos__)-1)/n_algos >= min_algos and algo_to_pop>-1: 
      bacc = acc
      algo_to_pop = -1
      for i in range(len(self.__algos__)):
        algo = self.__algos__.pop(0)
        tacc = self.validate_voting(x_val,y_val, method)
        if tacc > bacc:
          bacc = tacc
          algo_to_pop = i
        self.__algos__.append(algo)
      if algo_to_pop>-1:
        a = self.__algos__.pop(algo_to_pop)
        if verbose:
          print(f"Popped algo {algo_to_pop}: {a.name}")
    if verbose:
      print("Done removing algos")

  def validate_models(self, x_val=None, y_val=None, verbose=False):
    """Given validation data and labels, records the 
    validation accuracy of the models int he ensamble
    for later use with weighted voting. use 
    current_algos(), or current_algos_raw() in order 
    to see results."""
    _x,_y,x_val,y_val = self.__data_or_default(None,None,x_val=x_val,y_val=y_val)
    if x_val is None or y_val is None:
      raise ValueError("validation data is required for validate_models, either pass x_val= and y_val= args, or use \"set_default_data\"")
    
    for i in self.__algos__:
      tx,ty = i.transform(x_val, y_val)
      preds = i.predict_func(i.model, x_val)
      nright=0
      for p in len(preds):
        if preds[p]==y_val[p]:
          nright+=1
      i.val_score = nright/len(y_val)
