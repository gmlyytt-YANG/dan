import os
import numpy as np
from ImageServer import ImageServer
import time
import datetime
from OccluDetectionTraining import OccluDetectionTraining

datasetDir ="/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/data/"
content = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
trainSetName = "dataset_nimgs=62960.npz"
valSetName = "dataset_nimgs=1288.npz"

print("loading data")
trainOccluServer = ImageServer.Load(datasetDir + trainSetName)
valOccluServer = ImageServer.Load(datasetDir + valSetName)

print("network prepare")
training = OccluDetectionTraining()
training.loadData(trainOccluServer, valOccluServer)
training.initializeNetwork()

print("training")
training.train(0.001, num_epochs=100, networkDes=content)
