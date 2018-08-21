ModelInferencer Class
=====================

This class is useful for getting predictions for a trained model. This can be done by specifying a path to the data and having it pull random images to make predictions on, or by directly giving it images. This is useful to get a visual on what the model is outputting. 

Examples
--------
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
