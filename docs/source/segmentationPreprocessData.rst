Preprocess Data
===============

These are the functions that I used to preprocess that data that I had. It is assumed that the data directory specified by path is structured like this:
::
  dataDir
  |--class1
  |  |--data
  |  |--label
  |--class2
     |--data
     |--label

The primary functions that should be used directly from this file are renameLabels and preprocessData. The rename labels function should be used to make the label and image names match, and then the preprocess function should be used to split the images. Be aware that these functions leave the original data intact and only add train, validate, and test directories so the images will effectively take up twice the size. This was intended to that the preprocessData function could be run again to give images of separate sizes. After running the appropriate functions, the data of interest will be structured like so:
::
  dataDir
  |--class1
     |--subdir1
     |  |--data
     |  |--label
     |--subdir2
        |--data
        |--label

Example
-------
Here is an example of how the functions in this file should be used.

.. code-block:: python
  :linenos:

  # defining variables
  path = "../../../data/groundcover2016/"
  classNames = ['wheat','maize','maizevariety','mungbean']
  ignore = ["CE_NADIR_"]
  replace = {".tif" : ".jpg"}
  ignoreDirectories = ['train','validate','test']
  size = 5000
  shape = (256,256)
  trainProportion = 0.8
  validateProportion = 0.1

  # Rename and proprocess labels
  renameLabels(path,ignoreDirectories,ignore,replace)
  preprocessData(path,classNames,size,shape,trainProportion,validateProportion)

.. automodule:: segmentationPreprocessData
    :members:
    :undoc-members:
    :show-inheritance:
