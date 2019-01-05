import pandas as pd
import os
import numpy as np
from ImageServer import ImageServer
from FaceAlignment import FaceAlignment
from OccluServer import OccluServer

datasetDir ="/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/data/"

networkName = '/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/network/network-2018-12-20/network_00491_2018-12-20-14-17.npz'

nStages = 1
network = FaceAlignment(112, 112, 1, nStages=nStages)
network.loadNetwork(networkName)

# print ("Processing train set")
# trainSetPrefix = "dataset_nimgs=60960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112]"
# trainSetName = "trainOccluSet.npz"
# trainSet = ImageServer.Load(datasetDir + trainSetPrefix + "{}.npz".format(nStages - 1))
# trainOccluServer = OccluServer(network, trainSet)
# trainOccluServer.featureGenerating(trainLoad=True, normalized=True)
# trainOccluServer.Save(datasetDir, trainSetName)
# trainOccluServer = OccluServer.Load(datasetDir + trainSetName)
# # print (trainOccluServer.features.shape)
# # print (trainOccluServer.labels.shape)
# print ('-----------------')
# 
# print ("Processing val set")
# valSetPrefix = "dataset_nimgs=100_perturbations=[]_size=[112, 112]"
# valSetName = "valOccluSet.npz"
# valSet = ImageServer.Load(datasetDir + valSetPrefix + "{}.npz".format(nStages - 1))
# valOccluServer = OccluServer(network, valSet)
# valOccluServer.featureGenerating(normalized=True)
# valOccluServer.Save(datasetDir, valSetName)
# # valOccluServer = OccluServer.Load(datasetDir + valSetName)
# print ('-----------------')

print ("Processing common set")
commonSetPrefix = "commonSet"
commonSetName = "commonOccluSet.npz"
commonSet = ImageServer.Load(datasetDir + commonSetPrefix + "{}.npz".format(nStages - 1))
commonOccluServer = OccluServer(network, commonSet)
commonOccluServer.featureGenerating()
commonOccluServer.Save(datasetDir, commonSetName)
# commonOccluServer = OccluServer.Load(datasetDir + trainSetName)
print ('-----------------')

print ("Processing challenge set")
challengingSetPrefix = "challengingSet"
challengingSetName = "challengingOccluSet.npz"
challengingSet = ImageServer.Load(datasetDir + challengingSetPrefix + "{}.npz".format(nStages - 1))
challengingOccluServer = OccluServer(network, challengingSet)
challengingOccluServer.featureGenerating()
challengingOccluServer.Save(datasetDir, challengingSetName)
# challengingOccluServer = OccluServer.Load(datasetDir + challengingSetName)
print ('-----------------')

print ("Processing w300 set")
w300SetPrefix = "w300Set"
w300SetName = "w300OccluSet.npz"
w300Set = ImageServer.Load(datasetDir + w300SetPrefix + "{}.npz".format(nStages - 1))
w300OccluServer = OccluServer(network, w300Set)
w300OccluServer.featureGenerating()
w300OccluServer.Save(datasetDir, w300SetName)
# w300OccluServer = OccluServer.Load(datasetDir + w300SetName)
print ('-----------------')
