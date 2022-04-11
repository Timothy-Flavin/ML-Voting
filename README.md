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

### Init

The init_function should take four variables and return a train_score, val_score 
and a reference to an internal model that would need to be saved for future predictions.  

## Tunable Parameters

## Important Methods
