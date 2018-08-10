.. _Keras: https://keras.io/

Welcome to capstone-project's documentation!
============================================

.. toctree::
   :maxdepth: 2

   modules

This project indirectly uses a semantic segmentation neural network to segment images and determine the proportion of each class in a given image. The networks are built using Keras_ and are initially trained using the semantic segmentation network. Then a layer is added to the network so that the final output of the model, is the proportion of each class in the image. This is done by summing the number of pixels that the model predicts are in each class and dividing by the size of the image. Adding this extra layer turns further training into a regression problem. These regression networks can also be trained further to try and improve accuracy. Or models with this architecture could be trained fully via regression.

The data generators used to feed the models work on data structured like this:
::
  dataDir
  |--class1
     |--subdir1
     |  |--data
     |  |--label
     |--subdir2
        |--data
        |--label

The data for each class is placed into sub-directories, like shown above, to keep the directories with data in them somewhat small. This is done so that they still open quickly and also, to avoid loading millions of file names into RAM for the generators. The main files that would likely be usable to other people are the ModelTrainer class, the ModelInferencer class, and the segmentation preprocessing functions. Examples of how to use each of these classes and sets of functions are included in their documentation.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
