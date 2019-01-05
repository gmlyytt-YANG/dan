import pandas as pd
import os
import numpy as np
from ImageServer import ImageServer
from FaceAlignment import FaceAlignment
import tests

def computeMeanError(errors, clearErrors, occluErrors, occluNums, clearNums):
    avgError = np.mean(errors)
    occluError = occluErrors / float(occluNums)
    clearError = clearErrors / float(clearNums)

    return avgError * 100, clearError * 100, occluError * 100


datasetDir ="/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/data/"

verbose = False
showResults = False
showCED = True

normalization = 'centers'
failureThreshold = 0.08

# networkBase = '/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/network/network-2018-12-26-10-05'
networkBase = '/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/network/network-2018-12-20'
filenames = os.listdir(networkBase)
newFilenames = []
for filename in filenames:
    if os.path.splitext(filename)[-1] == '.npz':
        newFilenames.append(filename)

newFilenames.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
for filename in newFilenames:
    if filename != 'network_00491_2018-12-20-14-17.npz':
        continue
    nStages = 1
    network = FaceAlignment(112, 112, 1, nStages=nStages)
    network.loadNetwork(networkBase + '/' + filename)
    
    print (filename)
    # print ("Normalization is set to: " + normalization)
    # print ("Failure threshold is set to: " + str(failureThreshold))
    # print ('------------')
    
    # print ("Processing common subset of the 300W public test set (test sets of LFPW and HELEN)")
    commonSet = ImageServer.Load(datasetDir + "commonSet{}.npz".format(nStages - 1))
    # commonSet.baselineShow()
    commonErrs, commonOccluErrs, commonOccluNums, commonClearErrs, commonClearNums =\
        tests.LandmarkError(commonSet, network, normalization, showResults, verbose, datasetDir + 'commonSet{}.npz'.format(nStages), stage=nStages)
    commonAvgErr, commonClearError, commonOccluErr = computeMeanError(commonErrs, commonClearErrs, commonOccluErrs, commonOccluNums, commonClearNums)
    print(commonAvgErr, commonClearError, commonOccluErr)
    print ('------------')
    
    # print ("Processing challenging subset of the 300W public test set (IBUG dataset)")
    challengingSet = ImageServer.Load(datasetDir + "challengingSet{}.npz".format(nStages - 1))
    # challengingSet.baselineShow()
    challengingErrs, challengingOccluErrs, challengingOccluNums, challengingClearErrs, challengingClearNums =\
        tests.LandmarkError(challengingSet, network, normalization, showResults, verbose, datasetDir + 'challengingSet{}.npz'.format(nStages), stage=nStages)
    challengingAvgErr, challengingClearError, challengingOccluErr = computeMeanError(challengingErrs, challengingClearErrs, challengingOccluErrs, challengingOccluNums, challengingClearNums)
    # print(challengingAvgErr, challengingClearError, challengingOccluErr)
    # print('--------------')
    # 
    # print ("Processing 300W private test set")
    w300 = ImageServer.Load(datasetDir + "w300Set{}.npz".format(nStages - 1))
    # w300.baselineShow()
    w300Errs, w300OccluErrs, w300OccluNums, w300ClearErrs, w300ClearNums =\
        tests.LandmarkError(w300, network, normalization, showResults, verbose, datasetDir + 'w300SetSet{}.npz'.format(nStages), stage=nStages)
    # tests.AUCError(w300Errs, failureThreshold, showCurve=showCED)
    w300AvgErr, w300ClearError, w300OccluErr = computeMeanError(w300Errs, w300ClearErrs, w300OccluErrs, w300OccluNums, w300ClearNums)
    # print(w300AvgErr, w300ClearError, w300OccluErr)
    # print ('---------------')
    
    # print ("Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN")
    fullsetAvgErr = np.mean(commonErrs + challengingErrs)
    fullsetClearErr = (commonClearErrs + challengingClearErrs) / (commonClearNums + challengingClearNums)
    fullsetOccluErr = (commonOccluErrs + challengingOccluErrs) / (commonOccluNums + challengingOccluNums)
    # print ("Average error: {0}".format(np.mean(fullsetErrs) * 100))
    # print ("Clear error: {0}".format((commonClearErrs + challengingClearErrs) / (commonClearNums + challengingClearNums) * 100))
    # print ("Occlu error: {0}".format((commonOccluErrs + challengingOccluErrs) / (commonOccluNums + challengingOccluNums) * 100))
    # tests.AUCError(fullsetErrs, failureThreshold, showCurve=showCED)
    # print(np.mean(fullsetErrs), fullsetClearErr, fullsetOccluErr)
    # print ('----------------')
    
    errorShow = {
        'mean' : [commonAvgErr, challengingAvgErr, w300AvgErr, fullsetAvgErr * 100],
        'nclear': [commonClearError, challengingClearError, w300ClearError, fullsetClearErr * 100],
        'occlu': [commonOccluErr, challengingOccluErr, w300OccluErr, fullsetOccluErr * 100]
    }
    
    df = pd.DataFrame(errorShow, index=['common', 'challenge', '300w', 'fullset'])
    print(df)
    print('----------------------------------------')

