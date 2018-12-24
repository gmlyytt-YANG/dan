import numpy as np
from ImageServer import ImageServer
from FaceAlignment import FaceAlignment
import tests

datasetDir ="../data/"

verbose = False
showResults = False
showCED = True

normalization = 'centers'
failureThreshold = 0.08

networkFilename = '../network/network-2018-12-24-17-13/network_00494_2018-12-24-17-16.npz'
nStages = 2
network = FaceAlignment(112, 112, 1, nStages=nStages)
network.loadNetwork(networkFilename)

print ("Network being tested: " + networkFilename)
print ("Normalization is set to: " + normalization)
print ("Failure threshold is set to: " + str(failureThreshold))
print ('------------')

print ("Processing common subset of the 300W public test set (test sets of LFPW and HELEN)")
commonSet = ImageServer.Load(datasetDir + "commonSet{}.npz".format(nStages - 1))
commonSet.baselineShow()
commonErrs, commonOccluErrs, commonOccluNums, commonClearErrs, commonClearNums =\
    tests.LandmarkError(commonSet, network, normalization, showResults, verbose, datasetDir + 'commonSet{}.npz'.format(nStages), stage=nStages)
print ('------------')

print ("Processing challenging subset of the 300W public test set (IBUG dataset)")
challengingSet = ImageServer.Load(datasetDir + "challengingSet{}.npz".format(nStages - 1))
challengingSet.baselineShow()
challengingErrs, challengingOccluErrs, challengingOccluNums, challengingClearErrs, challengingClearNums =\
    tests.LandmarkError(challengingSet, network, normalization, showResults, verbose, datasetDir + 'challengingSet{}.npz'.format(nStages), stage=nStages)
print('--------------')

print ("Processing 300W private test set")
w300 = ImageServer.Load(datasetDir + "w300Set{}.npz".format(nStages - 1))
w300.baselineShow()
w300Errs, w300OccluErrs, w300OccluNums, w300ClearErrs, w300ClearNums =\
    tests.LandmarkError(w300, network, normalization, showResults, verbose, datasetDir + 'w300SetSet{}.npz'.format(nStages), stage=nStages)
tests.AUCError(w300Errs, failureThreshold, showCurve=showCED)
print ('---------------')

print ("Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN")
fullsetErrs = commonErrs + challengingErrs
print ("Average error: {0}".format(np.mean(fullsetErrs)))
print ("Clear error: {0}".format((commonClearErrs + challengingClearErrs) / (commonClearNums + challengingClearNums)))
print ("Occlu error: {0}".format((commonOccluErrs + challengingOccluErrs) / (commonOccluNums + challengingOccluNums)))
tests.AUCError(fullsetErrs, failureThreshold, showCurve=showCED)
print ('----------------')

# print ("Processing train set")
# trainSetPrefix = "dataset_nimgs=60960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112]"
# trainSet = ImageServer.Load(datasetDir + trainSetPrefix + "{}.npz".format(nStages - 1))
# trainSet.baselineShow()
# trainErrs, trainOccluErrs, trainOccluNums, trainClearErrs, trainClearNums =\
#     tests.LandmarkError(trainSet, network, normalization, showResults, verbose, datasetDir + trainSetPrefix + "{}.npz".format(nStages), train_load=True, normalized=True, stage=nStages)
# print ('-----------------')
# 
# print ("Processing val set")
# valSetPrefix = "dataset_nimgs=100_perturbations=[]_size=[112, 112]"
# valSet = ImageServer.Load(datasetDir + valSetPrefix + "{}.npz".format(nStages - 1))
# valSet.baselineShow()
# valErrs, valOccluErrs, valOccluNums, valClearErrs, valClearNums =\
#     tests.LandmarkError(valSet, network, normalization, showResults, verbose, datasetDir + valSetPrefix + "{}.npz".format(nStages), normalized=True, stage=nStages)

