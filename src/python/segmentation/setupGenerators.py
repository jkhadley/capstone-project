
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

#----------------------------------------------------------------------------------------------------------------
#                                             define paths to data
#----------------------------------------------------------------------------------------------------------------
# variables to tweak
epochs = 10



#----------------------------------------------------------------------------------------------------------------
#                                               Setup data generators
#----------------------------------------------------------------------------------------------------------------
def generateData(path, save_dir,batch_size):
    os.chdir(path)
    # Generator Parameters
    rot = 20
    width_shift = 0.1
    height_shift = 0.1
    target_size = (2048,1152)
    seed = 123

    if save_dir == False:
        data_aug_dir = None
        label_aug_dir = None
    else:
        data_aug_dir = "./train/augmentData"
        label_aug_dir = "./train/augmentLabel"

    data_datagen = ImageDataGenerator(
        #featurewise_center=True,
        rotation_range= rot,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        horizontal_flip=True#,
        #validation_split=0.15
        )

    image_generator = data_datagen.flow_from_directory(
        './train',
        classes = ["data"],
        class_mode = None,
        color_mode = "rgb",
        target_size = target_size,
        batch_size =batch_size,
        save_to_dir = data_aug_dir,
        save_prefix = "aug",
        seed = seed 
        )

    label_generator = data_datagen.flow_from_directory(
        './train',
        classes = ["label"],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = label_aug_dir,
        save_prefix = "aug",
        seed = seed 
        )

    # zip the generators together
    train_gen = zip(image_generator,label_generator)
    
    for (img,label) in train_gen:
        img,label = normalizeData(img,label)
        yield (img,label)

def normalizeData(image, label):
    if(np.max(image) > 1):
        image = image/255
    if(np.max(label) > 1):
        label = label/255
        label[label > 0.5] = 1
        label[label < 0.5] = 0

    return (image,label)
#----------------------------------------------------------------------------------------------------------------
#                                               test the generator
#----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    gen = generateData(True)
    
    for i in enumerate(gen):
        if(i >= 3):
            break
