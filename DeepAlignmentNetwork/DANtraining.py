from ImageServer import ImageServer
from FaceAlignmentTraining import FaceAlignmentTraining
import time
import datetime

datasetDir = "/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/data/"
content = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
networkFilename = "/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/network/network-2018-12-20/network_00491_2018-12-20-14-17.npz"
stages = 2
trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=60960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112]{}.npz".format(stages - 1))

validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=100_perturbations=[]_size=[112, 112]{}.npz".format(stages - 1))

#The parameters to the FaceAlignmentTraining constructor are: number of stages and indices of stages that will be trained
#first stage training only
# training = FaceAlignmentTraining(1, [0])
# second stage training only
training = FaceAlignmentTraining(stages, [stages - 1])

training.loadData(trainSet, validationSet)
training.initializeNetwork()

#load previously saved moved
training.loadNetwork(networkFilename)

training.train(0.0001, num_epochs=150, networkDes=content)
