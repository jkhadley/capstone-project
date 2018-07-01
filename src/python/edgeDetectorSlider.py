"""
Author(s): Joseph Hadley
Date Created:  2018-06-09
Date Modified: 2018-06-11
Description: Make a GUI with sliders to mess with the threshholds on the canny filter
"""
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication,QWidget,QGridLayout,QSlider,QGroupBox,QVBoxLayout,QLabel,QPushButton,QRadioButton
from copy import deepcopy
#-------------------------------------------------------------------------------------------------------------------
#                                                 GUI class
#-------------------------------------------------------------------------------------------------------------------
class EdgeDetectorApp(QWidget):
    
    def __init__(self):
        super().__init__()

        self.title = 'Edge Detector Application'
        self.left = 300
        self.top = 150
        self.width = 1200
        self.height = 800
        self.threshold0 = 200
        self.threshold1 = 100
        self.imageIndex = 0 
        self.numOfImages = len(os.listdir())       
        
        f = os.listdir()[0] # get a test image

        self.imageIndex = 0
        self.img = cv2.imread(f,1) # read in the image

        # initialize figures
        self.ogFigure = plt.figure()
        self.edgeFigure = plt.figure() 

        self.initUI()

    def initUI(self):
        # setup the window
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle(self.title)
        grid = QGridLayout()
        
        # create Sliders group boxs
        self.sliderGroup0 = self.createSlider('Threshold 1',self.threshold0)
        self.sliderGroup1 = self.createSlider('Threshold 2',self.threshold1)

        # extract the sliders out of the group boxes
        self.slider0 = self.sliderGroup0.findChild(QSlider)
        self.slider1 = self.sliderGroup1.findChild(QSlider)

        # declare what method to run when an update happens
        self.slider0.valueChanged.connect(self.thresholdChange)
        self.slider1.valueChanged.connect(self.thresholdChange)

        # initialize figure canvases
        self.ogCanvas = FigureCanvas(self.ogFigure)
        self.edgeCanvas = FigureCanvas(self.edgeFigure)

        # create image change buttons
        self.nextButton = QPushButton("Next")
        self.prevButton = QPushButton("Prev")

        # link buttons to methods
        self.nextButton.clicked.connect(self.nextImage)
        self.prevButton.clicked.connect(self.prevImage)

        # create radio buttons to change the colors
        self.rgb = QRadioButton("RGB")
        self.hsl = QRadioButton("HLS")
        self.ycc = QRadioButton("YCrCb")

        # make a groupbox to convert the color layers


        # place buttons to change image
        grid.addWidget(self.nextButton,0,3,1,1)
        grid.addWidget(self.prevButton,0,0,1,1)

        # place canvas's
        grid.addWidget(self.ogCanvas,1,0,2,2)
        grid.addWidget(self.edgeCanvas,1,2,2,2)

        # place sliders (arg order, object, row, column, row span, column span)
        grid.addWidget(self.sliderGroup0,3,0,1,4)
        grid.addWidget(self.sliderGroup1,4,0,1,4)

        # Update the Images
        self.plotOriginal()
        self.updateEdge()

        self.setLayout(grid)
        # show gui
        self.show()
    
    def createSlider(self,title,initialValue):
        # method to create a slider
        groupBox = QGroupBox(title)

        slider = QSlider(Qt.Horizontal)
        slider.setTickPosition(2)
        slider.setTickInterval(50)
        slider.setMinimum(0)
        slider.setMaximum(300)
        slider.setSingleStep(10)
        slider.setValue(initialValue)

        vbox = QVBoxLayout()
        vbox.addWidget(slider)        
        vbox.addStretch(2)
        groupBox.setLayout(vbox)

        return groupBox
    def plotOriginal(self):
        self.ogFigure.clear() # remove old graph
        
        ax = self.ogFigure.add_subplot(111) # define axis to plot on

        ax.imshow(self.img)
        
        ax.set_title("Original Image")
        ax.axis("off")

        self.ogCanvas.draw() # refresh the canvas
    
    def updateEdge(self):
        self.edgeFigure.clear() # remove old graph
        ax = self.edgeFigure.add_subplot(111) # define axis to plot on

        edge = cv2.Canny(self.img,self.threshold0,self.threshold1)
        ax.imshow(edge)
        ax.set_title("Image Edges")
        ax.axis("off")

        self.edgeCanvas.draw() # refresh the canvas

    def thresholdChange(self):
        # get the values from the slider
        self.threshold0 = self.slider0.value()
        self.threshold1 = self.slider1.value()   
        # update the edge image
        self.updateEdge()
    
    def nextImage(self):
        if self.imageIndex < self.numOfImages:
            self.imageIndex -= 1
        else:
            self.imageIndex = self.numOfImages

        self.img = cv2.imread(os.listdir()[self.imageIndex],1) # read in the image
        self.plotOriginal()
        self.updateEdge()

    def prevImage(self):
        
        if self.imageIndex > 0:
            self.imageIndex -= 1
        else:
            self.imageIndex = 0

        self.img = cv2.imread(os.listdir()[self.imageIndex],1) # read in the image
        self.plotOriginal()
        self.updateEdge()

#-------------------------------------------------------------------------------------------------------------------
#                                                     Run the program
#-------------------------------------------------------------------------------------------------------------------
# run program if it is the script being run directly
if __name__ == "__main__":
    # Change directory to one with images
    os.chdir("../../data/groundcover2016/maize/data")   
    app = QApplication(sys.argv)
    edgeDetector = EdgeDetectorApp()
    sys.exit(app.exec_())

