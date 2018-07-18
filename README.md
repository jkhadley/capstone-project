## Introduction
For my capstone project, I have two sets of data. One set consists of images of corn, wheat, and mungbeans. For this data set, the goal is to find the area of the leaves in the picture. To do this semantic segmentation will be used. Since there are not any reference points in these images, there is no way to tell how close the leaf is to the lens. Therefore, there is no real way to accurately calculate the areas of the leaves and therefore the objective was to calculate the proportion of the image that the leaf was taking up.

This data set consists of small, uniformly sized images of corn, that were split into 4 categories. These categories are healthy or having 1 of 3 diseases. The goal with this data set is to build a classifier that will classify the images as being healthy or having a disease. This problem will be solved using a Convolutional Neural Network.


## Segmentation
<div style="text-align: justify">
To start simple, 
For The Image segmentation, the first architecture that was used was [U-net](https://arxiv.org/abs/1505.04597). Since segmentation is done without fully connected layers, images of multiple sizes can be fed to the network as long as the network was trained on larger images. Because the networks have flexible inputs, I started training the networks with images of size 2048 x 1152 since they were the most popular size in my data set. Training the network on images of this size was problematic because the number of parameters created due to using images of this size was very large and in order to fit the model and one image to train on in to RAM, the depth of the convolutional layers used was significantly less than the paper linked above used. Due to the shallowness of the convolutional layers, the model was not able to learn that well and only resulted in accuracies of about 88%. In order to get the batch size up and number of parameters in the model down, the training images where then cropped to sizes of 256 x 256 and the model was built around that input and output sizes. This model yielded about 97% accuracy. Some outputs of the smaller network can be seen below.
</div>

<div style="text-align: justify">
As can be seen in the test output shown above, the model predicts all of the vegetation as well as the shadows near the corn plants as being corn. I believe
that this is partially due to the fact the a sigmoid was used for the output activation and could likely be tweaked to give better results. I believe that in
order to get better results, the model should be trained on all the classes so that the model can learn that there are more states than just plant and not-plant. To test this hypothesis, I am working on setting up the images and generators in Keras to do multi-class segmentation.  
</div>

## Classification
<div style="text-align: justify">
This data set consists of small, uniformly sized images of corn, that were split into 4 categories. These categories are healthy or having 1 of 3 diseases. The goal with this data set is to build a classifier that will classify the images as being healthy or having a disease. This problem will be solved using a Convolutional Neural Network.
</div>

### Binary Classification
<div style="text-align: justify">
To start this problem, we will first see how well a model can do at classifying if the images are healthy, or if they have a disease. Since each category has about the same amount of images, the data would be imbalanced if all of the data was used to train the model. To avoid the problems that can be caused by imbalanced data, the diseased leafs will be undersampled so that the amount of data is equal for the healthy and unhealthy leaf classes.
</div>

### Multi-Class Classification
Once it is found that the binary classification works well, I will move on to trying to predict whether the plant is healthy or if it has one of the three diseases mentioned.

## Moving Forward
Although the goal of the segmentation is to try and predict the area of the leaves, I believe that these two projects could be combined and modified and put into an application where you could take a picture of some plant, have the segmentation network get an idea of what plants are in the network, and then use a modified version of the leaf health classifier to determine whether or not the plant in the image is healthy or has a disease. This application could then show the user which part of the image or plant in question has the disease. This project could be used with technology such as drones in order to better monitor the health of crops.
