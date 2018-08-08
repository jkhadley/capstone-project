.. _Keras: https://keras.io/

Welcome to capstone-project's documentation!
============================================
.. toctree::
   :maxdepth: 2

   modules

This project indirectly uses a semantic segmentation neural network to segment images and determine the proportion of each class in a given image. The networks built using Keras_ and are initially trained using the semantic segmentation network and then a layer is added to the network so that the final output is the proportion of each class in the image. This is done by summing the number of pixels that the model predicts are in each class and then by dividing this by the size of the image. Adding this extra layer turns further training into a regression problem. These regression networks can also be trained further to try and improve accuracy.

The data generators work on data structured like:
::
  dataDir
  |--class1
  |  |--subdir1
  |  |  |--data
  |  |  |--label
  |  |--subdir2
  |  |  |--data
  |  |  |--label

The data for each class is placed into subdirectories to keep the directories with data in them somewhat small, so that they still open quickly and to avoid loading millions of file names into RAM for the generators.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
