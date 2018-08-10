ModelInferencer Class
=====================

This class is useful for getting predictions for a trained model. This can be done by specifying a path to the data and having it pull random images to make predictions on, or by directly giving it images. This is useful to get a visual on what the model is outputting. An example of using this class to make predictions on a batch of random images can be seen here:

.. code-block:: python
  :linenos:

  classMap = {'class1' : 1,
              'class2' : 2}

  m = ModelInferencer("model/to/load",data_path = "path/to/data")
  m.setClassMap(classMap)
  m.setBatchSize(5)

  # make predictions
  m.batchPredict()

An example of using this class to make a prediction on a specific image is shown here:

.. code-block:: python
  :linenos:
  
  from skimage import io

  # load an image
  image = io.imread("image/to/load")

  m = ModelInferencer("model/to/load")
  output = m.predict(image)

.. automodule:: modelInferencer
    :members:
    :undoc-members:
    :show-inheritance:
