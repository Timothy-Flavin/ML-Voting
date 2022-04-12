# ML-Voting
ML_Democracy is an Ensamble voting pipeline made for creating an ensamble of models from different libraries. 
For example, an ensamble of neural networks including both Tensorflow CNN and Sklearn feed forward networks.
This Library works by using one of several weighted voting methods to combine ML models in as lightweight a way as possible.
Each model is required to meet a few criteria in order to be used in ML_Democracy.

## Citizenship Requirements

Algorithms which belong to ML_Democracy must an instance of the "ML_Algo" object and they must be added 
to the ML_Democracy using "add_algo". Each ML_Algo needs an init_function, a predict function, a name, 
and a transformation. 

### Name

Each model will have a name string used for editing the ensamble and viewing results. This name
will be unique in order to access individual models in ML_Democracy. When models are added using
the (num>=2) argument, a number will be appened to the name string for each model counting up from 
zero. For example: 

```
model.name = "svm"
add_algo(model, num=3, verbose=True)

>>> Added: svm 0
>>> Added: svm 1
>>> Added: svm 2
```

### Transformation

Different models expect different shaped inputs. In order to account for this, each model will have a transformation 
method which takes an x and a y argument where x is a numpy array of data and y is a numpy array of labels. If no
Transformation is given, a default function pointer which simply returns x and y will be used. 
The transformation must return a transformed x and y, and it must be able to handle a null y argument in the case
of transforming unlabeled data. 

### Feature Sampling

One staple of the Random forest algorithm, and other ensamble voting methods, is the ability to expose only some of 
the data features to each member of the ensamble. ML_Democracy can do this in one of two ways, first: random 
selection of a proportion of the features where featureProportion is within \[0.0,1.0\] inclusive. A negative value 
will cause no feature sampling, so that each model will get the full dataset. Another way to sample features is to 
select a contiguous section of the data, such as a number of rows in an image. This method can be used for when 
spatial relations are important and it can be enabled by setting the "fast" argument to True. Fast refers to the 
fast trees algorithm which uses such a sampling method. The argument *axis=-1* refers to which axis to sample on. 
For example, a dataset of points in 3 dimensional space may be stored as a matrix where each row is a point and
each column is a dimension. In this case one may want each learner to see only two dimensions at a time, so an 
axis of -1 would cause the sampling to occur on the features, the dimensions. In another example, data might be 
a series of images in which each instance is a two dimensional array of data. In this case, one may want to sample 
rows of the image or columns and could therefor chose an axis of 1 or 2 respectively. 

### Init: "fit_func"

The init_function should take four variables and return a train_score, val_score, and model pointer
and a reference to an internal model that would need to be saved for future predictions. The init function
should create a model, train it, and optionally validate the model. If no validation set is used it may
return a validation score of "None" but this will cause some of the weighting methods to be disabled. Some
Weightings rely on a validation score and so these methods will throw an error on models with no validation
scores. One example of an init function can be seen below:

```
def model_init(x_train, y_train, x_val, y_val):
  model = Classifier().fit(x_train, y_train)
  test_score = model.score(x_val, y_val)
  train_score = model.score(x_train, y_train)
  return train_score, val_score, model 
```

### Predict

Each model will also need a predict function which takes an array of datapoints that returns a 1D array of
labels. For Sklearn models this is very simple:
```
def predict(model, x):
  temp = model.predict(x)
  return temp
```
For models which use softmax or something similar, this function will need to return the index of the max
output rather than an array of prediction weights for each datapoint in x. Predict will be called after the
Transform method in all instances, This can be used for more than intended, because the pipeline is: 

```
def train_algos(x,y,...)
  for m in models:
  ... bootsrap and feature sampling ...
  x,y = m.Transform(x,y)
  val_x,val_y = m.Transform(val_x,val_y)
  train_score,val_score,model = m.fit_func(x,y,val_x,val_y)
```

in all cases. As long as the supplied Tansform, fit, and predict functions supplied work together, ML_Democracy 
will not care what happens to the data. 

## Tunable Parameters

bag: True or False, whether bootstrap sampling 

fast: True or False, Whether features sampled are contiguous

verbose: True or False, Whether to print what is happening or stay quiet

x_train, y_train, x_val, y_val: Data to be used for fitting. If none is supplied, ML_Democracy can also be initialized
with a default_data argument which will be used instead. 

pre_trained: Each ML_Algo object has a pre_trained parameter which tracks whether the model has been trained or not already.
If this is set to true, this model will not be trained in the "train_algos" function. Good for models that take a long time
to train, but can work with other models. 

## Future work

- Custom Saving method for models to make saving ensambles and editing them later easy
- Boosting
- Reimplement Stacking
- Pip installation
- Allow arg bindings in fit_func so that more customizable initializations can be used
- Graphing metrics
- Auto-prunning

