lala
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.integrate import simps
from matplotlib import pyplot as plt

def LandmarkError(imageServer, faceAlignment, normalization='centers', showResults=False, verbose=False):
    errors = []
    nImgs = len(imageServer.imgs)
    # print(nImgs)
    output_dir = './tmp2'
    # print(len(imageServer.newFilenames))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(nImgs):
        # if i > 200:
        #     break
        initLandmarks = imageServer.initLandmarks[i]
        gtLandmarks = imageServer.gtLandmarks[i]
        img = imageServer.imgs[i]
        # print(img)
        prefix = os.path.splitext('_'.join(imageServer.newFilenames[i].split('/')[3:]))[0]
        # print(prefix)
        
        if img.shape[0] > 1:
            img = np.mean(img, axis=0)[np.newaxis]

        resLandmarks = initLandmarks
        resLandmarks = faceAlignment.processImg(img, resLandmarks, normalized=True)
        img = img.transpose((1, 2, 0))
        print(prefix)
        cv2.imwrite(os.path.join(output_dir, prefix+'.jpg'), img)
        np.savetxt(os.path.join(output_dir, prefix + '.pts'), gtLandmarks)
        np.savetxt(os.path.join(output_dir, prefix + '.rpts'), resLandmarks)
        
        # print(resLandmarks)
        # print(gtLandmarks)
        # print('-------')
        # cv2.imwrite(filename, img.transpose((1,2,0)))
        # rptsPath = os.path.splitext(filename)[0] + '.rpts'
        # np.savetxt(rptsPath, resLandmarks)  
        
        if normalization == 'centers':
            normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
        elif normalization == 'corners':
            normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
        elif normalization == 'diagonal':
            height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
            normDist = np.sqrt(width ** 2 + height ** 2)
        
        error = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks)**2,axis=1))) / normDist       
        errors.append(error)
        if verbose:
            print("{0}: {1}".format(i, error))

        if showResults:
            plt.imshow(img[0], cmap=plt.cm.gray)            
            plt.plot(resLandmarks[:, 0], resLandmarks[:, 1], 'o')
            plt.show()

    if verbose:
        print "Image idxs sorted by error"
        print np.argsort(errors)
    avgError = np.mean(errors)
    print "Average error: {0}".format(avgError)

    return errors


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

    
