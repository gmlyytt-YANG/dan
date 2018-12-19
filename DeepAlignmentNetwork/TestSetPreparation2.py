from ImageServer import ImageServer
import numpy as np

trainValImageDirs = ["../data/images/lfpw/trainset/", "../data/images/helen/trainset/", "../data/images/afw/"]
trainValBoundingBoxFiles = ["../data/boxesLFPWTrain.pkl", "../data/boxesHelenTrain.pkl", "../data/boxesAFW.pkl"]

datasetDir = "../data/"

meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]

trainValSet = ImageServer(initialization='box')
trainValSet.PrepareData(trainValImageDirs, trainValBoundingBoxFiles, meanShape, 0, 10000,False)
trainValSet.LoadImages()
trainValSet.CropResizeRotateAll()
trainValSet.imgs = trainValSet.imgs.astype(np.float32)
trainValSet.Save(datasetDir, "trainValSet.npz")
