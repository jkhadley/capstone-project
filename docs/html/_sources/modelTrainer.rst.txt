ModelTrainer Class
==================

This is the main class that I would expect to be used. It uses the generators, metrics, and callbacks files to work and is used to train and evaluate segmentation and regression models. Some example of how to use it can be seen in the examples section.

Examples
--------
Here is an example of how to create model and train it.

.. code-block:: python
  :linenos:

  # initalize variables
  data_path    = "/path/to/data/"
  model_path   = "/path/to/models/"
  results_path = "/path/to/results/"

  classMap = {
    'class1' : 1,
    'class2': 2,
    'class3': 3
  }

  # Create model trainer
  modelTrainer = ModelTrainer(data_path,results_path,model_path)

  # Set Parameters
  modelTrainer.changeMetrics(['acc',recall,precision,f1Score])
  modelTrainer.changeBatchSize(64)
  modelTrainer.setClassMap(classMap)
  modelTrainer.setWeightInitializer('he_normal')
  modelTrainer.setOptimizerParams(lr = 1.0*(10**-3),momentum = 0.8,decay = 1.0*(10**-8))
  modelTrainer.changeDropout(0.6)
  modelTrainer.setSaveName("model1")

  # train the model
  modelTrainer.train()

Here is an example of loading a trained model and evaluating it on the training and validation data:

.. code-block:: python
  :linenos:

  # initalize variables
  data_path    = "/path/to/data/"
  model_path   = "/path/to/models/"
  results_path = "/path/to/results/"

  classMap = {
    'class1' : 1,
    'class2': 2,
    'class3': 3
  }

  # Create model trainer
  modelTrainer = ModelTrainer(data_path,results_path,model_path)

  # Set Parameters
  modelTrainer.changeMetrics(['acc',recall,precision,f1Score])
  modelTrainer.changeBatchSize(64)
  modelTrainer.setClassMap(classMap)
  modelTrainer.setOldModel('model1')
  # evaluate the model
  modelTrainer.evaluate()


.. automodule:: modelTrainer
    :members:
    :undoc-members:
    :show-inheritance:
