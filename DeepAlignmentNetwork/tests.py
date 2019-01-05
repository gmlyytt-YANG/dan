import os
import numpy as np
import cv2
import shutil
from matplotlib import pyplot as plt
from scipy.integrate import simps
from matplotlib import pyplot as plt


def LandmarkError(imageServer, faceAlignment, normalization='centers', showResults=False, verbose=False, saveFilename="", train_load=False, normalized=False, stage=1):
    errors = []
    nImgs = len(imageServer.imgs)
    roughLandmarks = []
    # print(nImgs)
    # print(len(imageServer.newFilenames))
    # output_dir = '../data/roughFaceAlignment'
    # imageServer.roughLandmarks = []
    occluErrors = 0.0
    clearErrors = 0.0
    occluNums = 0

    for i in range(nImgs):
        # if i > 200:
        #     break
        initLandmarks = imageServer.initLandmarks[i]
        gtLandmarks = imageServer.gtLandmarks[i]
        img = imageServer.imgs[i]
        # print(imageServer.filenames[i])
        roughLandmark = imageServer.roughLandmarks[i]
        # print(img)
        if train_load:
            prefix = os.path.splitext(imageServer.newFilenames[i].split('/')[-1])[0]
        else:
            prefix = os.path.splitext(imageServer.filenames[i].split('/')[-1])[0]
        # print(prefix)
        
        if img.shape[0] > 1:
            img = np.mean(img, axis=0)[np.newaxis]

        resLandmarks = initLandmarks
        resLandmarks = faceAlignment.processImg(img, resLandmarks, normalized=normalized)
        
        # img = img.transpose((1, 2, 0))
        # cv2.imwrite(os.path.join(output_dir, prefix + '.jpg'), img)
        # np.savetxt(os.path.join(output_dir, prefix + '.pts'), gtLandmarks)
        # np.savetxt(os.path.join(output_dir, prefix + '.rpts'), resLandmarks)
        
        # print(gtLandmarks.shape)
        
        if normalization == 'centers':
            normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
        elif normalization == 'corners':
            normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
        elif normalization == 'diagonal':
            height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
            normDist = np.sqrt(width ** 2 + height ** 2)
        
        if stage > 1:
            weight = {'occlu': 1.0, 'clear': 1.0}
            b = weight['clear']
            a = weight['occlu'] - weight['clear']

            occlu = roughLandmark[272:340]
            scaledOcclu = np.reshape(1 / (a * occlu + b), (68, 1))
            firstStageLandmarks = np.reshape(roughLandmark[136:272], (68, 2))
            transformedLandmarks = (resLandmarks - firstStageLandmarks) * scaledOcclu + firstStageLandmarks
            landmarkError = np.sqrt(np.sum((gtLandmarks - transformedLandmarks)**2,axis=1)) / normDist
            # print(scaledOcclu)
            # print(resLandmarks)
            # print(firstStageLandmarks) 
            # print(transformedLandmarks)
            # print(normDist)
            # print(landmarkError)
            # print('-------------')
        else:
            landmarkError = np.sqrt(np.sum((gtLandmarks - resLandmarks)**2,axis=1)) / normDist
        error = np.mean(landmarkError)  
        errors.append(error)
        
        # if stage > 1:
        #     roughLandmark = np.hstack((gtLandmarks.flatten(), (resLandmarks + firstStageLandmarks).flatten(), imageServer.occlus[i], landmarkError))
        if stage == 1:
            roughLandmark = np.hstack((gtLandmarks.flatten(), resLandmarks.flatten(), imageServer.occlus[i], landmarkError))
        # print(roughLandmark.shape)
        # print(roughLandmark)
        roughLandmarks.append(roughLandmark)
        occluError = np.dot(imageServer.occlus[i], landmarkError)
        
        # print(occluError)
        occluErrors += occluError
        occluNums += np.sum(imageServer.occlus[i])
        
        clearError = np.dot(1 - imageServer.occlus[i], landmarkError)
        # print(clearError)
        # print('-----------')
        clearErrors += clearError
        if verbose:
            print("{0}: {1}".format(i, error))

        if showResults:
            plt.imshow(img[0], cmap=plt.cm.gray)            
            plt.plot(resLandmarks[:, 0], resLandmarks[:, 1], 'o')
            plt.show()
        
        # if i > 5:
        #     break

    imageServer.roughLandmarks = np.array(roughLandmarks)

    if verbose:
        print "Image idxs sorted by error"
        print np.argsort(errors)
    
    avgError = np.mean(errors)
    occluError = occluErrors / float(occluNums)
    clearError = clearErrors / float(68 * len(imageServer.imgs) - occluNums)

    # print("Average error: {0}".format(avgError * 100))
    # print("Clear error: {}".format(clearError * 100))
    # print("Occlu error: {}".format(occluError * 100))
   
    # print(saveFilename)
    arrays = {key:value for key, value in imageServer.__dict__.items() if not key.startswith('__') and not callable(key)}
    np.savez(saveFilename, **arrays)
    
    # return errors, occluErrors, occluNums, clearErrors, (68 * len(imageServer.imgs) - occluNums)
    return errors, occluErrors, occluNums, clearErrors, (68 * len(imageServer.imgs) - occluNums)


def AUCError(errors, failureThreshold, step=0.0001, showCurve=False):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))

    ced =  [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    print "AUC @ {0}: {1}".format(failureThreshold, AUC)
    print "Failure rate: {0}".format(failureRate)

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    
