import os
from scipy import ndimage
import cv2
import numpy as np
import utils
import cPickle as pickle
import glob
from os import path

class ImageServer(object):
    def __init__(self, imgSize=[112, 112], frameFraction=0.25, initialization='box', datasetDir=None, color=False):
        self.origLandmarks = []
        self.roughLandmarks = []
        self.filenames = []
        self.mirrors = []
        self.meanShape = np.array([])

        self.meanImg = np.array([])
        self.stdDevImg = np.array([])

        self.perturbations = []

        self.imgSize = imgSize
        self.frameFraction = frameFraction
        self.initialization = initialization
        self.datasetDir = datasetDir
        self.color = color

        self.boundingBoxes = []

    @staticmethod
    def Load(filename):
        imageServer = ImageServer()
        arrays = np.load(filename)
        imageServer.__dict__.update(arrays)

        if (len(imageServer.imgs.shape) == 3):
            imageServer.imgs = imageServer.imgs[:, np.newaxis]

        return imageServer

    def Save(self, filename=None):
        if filename is None:
            # filename = "dataset_nimgs={0}_perturbations={1}_size={2}".format(len(self.imgs), list(self.perturbations), self.imgSize)
            filename = "dataset_nimgs={0}".format(len(self.imgs))
            if self.color:
                filename += "_color={0}".format(self.color)
            filename += ".npz"

        arrays = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        np.savez(self.datasetDir + filename, **arrays)

    def PrepareData(self, imageDirs, boundingBoxFiles, meanShape, startIdx, nImgs, mirrorFlag):
        filenames = []
        landmarks = []
        boundingBoxes = []
       
        
        for i in range(len(imageDirs)):
            filenamesInDir = glob.glob(imageDirs[i] + "*.jpg")
            filenamesInDir += glob.glob(imageDirs[i] + "*.png")
            # print(imageDirs[i], len(filenamesInDir))  

            if boundingBoxFiles is not None:
                boundingBoxDict = pickle.load(open(boundingBoxFiles[i], 'rb'))

            for j in range(len(filenamesInDir)):
                filenames.append(filenamesInDir[j])
                # print(filenamesInDir[j])
                ptsFilename = filenamesInDir[j][:-3] + "opts"
                landmarks.append(utils.loadFromOPts(ptsFilename))

                if boundingBoxFiles is not None:
                    basename = path.basename(filenamesInDir[j])
                    boundingBoxes.append(boundingBoxDict[basename])
                

        filenames = filenames[startIdx : startIdx + nImgs]
        landmarks = landmarks[startIdx : startIdx + nImgs]
        boundingBoxes = boundingBoxes[startIdx : startIdx + nImgs]
        # print(len(filenames))
        mirrorList = [False for i in range(nImgs)]
        if mirrorFlag:     
            mirrorList = mirrorList + [True for i in range(nImgs)]
            filenames = np.concatenate((filenames, filenames))

            landmarks = np.vstack((landmarks, landmarks))
            boundingBoxes = np.vstack((boundingBoxes, boundingBoxes))       

        self.origLandmarks = np.array(landmarks)
        self.filenames = filenames
        self.mirrors = mirrorList
        self.meanShape = meanShape
        self.boundingBoxes = boundingBoxes

    def LoadImages(self):
        self.imgs = []
        self.initLandmarks = []
        self.gtLandmarks = []
        self.occlus = []
        
        origLandmarks = self.origLandmarks.copy()
        # print(len(self.filenames))
        for i in range(len(self.filenames)):
            img = cv2.imread(self.filenames[i])
            # pint(img.shape)
            groundTruth = origLandmarks[i][:, :2]
            box = None if len(self.boundingBoxes) <= 2 else self.boundingBoxes[i]
            if len(self.boundingBoxes) > 2:
                self.boundingBoxes[i] = box
            img, groundTruth, box = utils.scaleImgLandmark(img, box, groundTruth)
            filename = os.path.splitext(self.filenames[i])[0].split('/')[-1]
            origLandmarks[i][:, :2] = groundTruth
            # if not train:
            #     print(img.shape)
            #     print('----------')
            #     cv2.imwrite('./tmp/' + self.filenames[i].split('/')[-1], img)
            if self.color:
                if len(img.shape) == 2:
                    img = np.dstack((img, img, img))
            else:
                if len(img.shape) > 2:
                    img = np.mean(img, axis=2)
            img = img.astype(np.uint8)
            occlu = origLandmarks[i][:, -1].astype(np.int)
            if self.mirrors[i]:
                # print(img.shape)
                # print(origLandmarks[i])
                allLandmarks = utils.mirrorShape(origLandmarks[i], img.shape)
                # print(allLandmarks)
                # print('-------------------')
                origLandmarks[i] = allLandmarks
                occlu = allLandmarks[:, -1].astype(np.int)    
                # print(occlu)
                img = np.fliplr(img)

            if self.color:
                img = np.transpose(img, (2, 0, 1))
            else:
                img = img[np.newaxis]

            groundTruth = origLandmarks[i][:, :2]
            # print(groundTruth)
            # print(occlu)
            if self.initialization == 'rect':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape)
            elif self.initialization == 'similarity':
                bestFit = utils.bestFit(groundTruth, self.meanShape)
            elif self.initialization == 'box':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape, box=self.boundingBoxes[i])
            
            # cv2.imwrite("../data/roughFaceAlignment/{}.png".format(filename), np.transpose(img, (1, 2, 0)))
            # np.savetxt("../data/roughFaceAlignment/{}.npz".format(filename), bestFit)
            # np.savetxt("../data/roughFaceAlignment/{}.pnpz".format(filename), groundTruth)

            self.imgs.append(img)
            self.initLandmarks.append(bestFit)
            self.gtLandmarks.append(groundTruth)
            self.occlus.append(occlu)
        
        self.origLandmarks = np.array(origLandmarks)
        self.initLandmarks = np.array(self.initLandmarks)
        self.gtLandmarks = np.array(self.gtLandmarks)    
        self.occlus = np.array(self.occlus)    

    def GeneratePerturbations(self, nPerturbations, perturbations):
        self.perturbations = perturbations
        # print(self.meanShape)
        # print(self.meanShape.max(axis=0))
        # print(self.meanShape.min(axis=0))
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)
        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        newImgs = []  
        newGtLandmarks = []
        newInitLandmarks = []
        newFilenames = []
        
        newOccluImgs = []
        newOcclus = []
        newOccluGtLandmarks = []
        newOccluInitLandmarks = []
        newOccluFilenames = []

        occlus = []
        
        # newBoundingBoxes = []

        translationMultX, translationMultY, rotationStdDev, scaleStdDev = perturbations

        rotationStdDevRad = rotationStdDev * np.pi / 180         
        translationStdDevX = translationMultX * (scaledMeanShape[:, 0].max() - scaledMeanShape[:, 0].min())
        translationStdDevY = translationMultY * (scaledMeanShape[:, 1].max() - scaledMeanShape[:, 1].min())
        print "Creating perturbations of " + str(self.gtLandmarks.shape[0]) + " shapes"

        # boundingBoxes = self.boundingBoxes.copy() 
        # count = 0
        for i in range(self.initLandmarks.shape[0]):
            if (i + 1) % 100 == 0:
                utils.logger('processed {} imgs'.format(i + 1))
            # boundingBox = np.reshape(self.boundingBoxes[i], (2, 2))
            # print(boundingBox)
            for j in range(nPerturbations):
                tempInit = self.initLandmarks[i].copy()
                # print(tempInit)
                angle = np.random.normal(0, rotationStdDevRad)
                offset = [np.random.normal(0, translationStdDevX), np.random.normal(0, translationStdDevY)]
                scaling = np.random.normal(1, scaleStdDev)

                R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])     
            
                tempInit = tempInit + offset
                tempInit = (tempInit - tempInit.mean(axis=0)) * scaling + tempInit.mean(axis=0)            
                tempInit = np.dot(R, (tempInit - tempInit.mean(axis=0)).T).T + tempInit.mean(axis=0)

                tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], tempInit, self.gtLandmarks[i])
                cv2.imwrite(self.datasetDir + "roughFaceAlignment/{}_{}.png".format(i, j), np.transpose(tempImg, (1, 2, 0)))
                # np.savetxt(self.datasetDir + "roughFaceAlignment/{}_{}.npz".format(i, j), tempInit)
                np.savetxt(self.datasetDir + "roughFaceAlignment/{}_{}.pnpz".format(i, j), tempGroundTruth)
                np.savetxt(self.datasetDir + "roughFaceAlignment/{}_{}.npz".format(i, j), self.occlus[i])
                # tempImg, tempInit, tempGroundTruth, tempBbox = self.CropResizeRotate2(self.imgs[i], boundingBox, tempInit, self.gtLandmarks[i]) 
                # for elem in tempGroundTruth:
                #     for _ in elem:
                #         if _ < 0:
                #             print(_)
                
                # tempImg = np.transpose(tempImg, (1, 2, 0))
                # cv2.imwrite("tmp/{}.jpg".format(count), tempImg)
                # np.savetxt("tmp/{}.npz".format(count), tempBbox)
                # if np.sum(self.occlus[i]) > 0:
                # print(tempImg.shape)
                newfilename = (self.filenames[i][:-4] + '_' + str(i) + '_' + str(j)).split('/')[-1]
                if np.sum(self.occlus[i]) > 0:
                    # print(self.occlus[i])
                    noisedTempImg = utils.gaussian_noise(tempImg, True)
                    cv2.imwrite(self.datasetDir + "roughFaceAlignment/{}_{}_noised.png".format(i, j), np.transpose(tempImg, (1, 2, 0)))
                    # np.savetxt(self.datasetDir + "roughFaceAlignment/{}_{}_noised.npz".format(i, j), tempInit)
                    np.savetxt(self.datasetDir + "roughFaceAlignment/{}_{}_noised.pnpz".format(i, j), tempGroundTruth)
                    np.savetxt(self.datasetDir + "roughFaceAlignment/{}_{}_noised.npz".format(i, j), self.occlus[i])
                    # tempImg = np.transpose(tempImg, (1, 2, 0))
                    # noisedTempImg = np.transpose(noisedTempImg, (1, 2, 0))
                    # cv2.imwrite("tmp/{}_ori.jpg".format(count), tempImg)
                    # cv2.imwrite("tmp/{}_noise.jpg".format(count), noisedTempImg)
                    # print(noisedTempImg.shape)
                    # print(noisedTempImg - tempImg)
                    # print('-------------')
                    newOccluImgs.append(noisedTempImg)
                    newOcclus.append(self.occlus[i])
                    newOccluInitLandmarks.append(tempInit)
                    newOccluGtLandmarks.append(tempGroundTruth)
                    newOccluFilenames.append(newfilename)

                newImgs.append(tempImg)
                newInitLandmarks.append(tempInit)
                newGtLandmarks.append(tempGroundTruth)
                newFilenames.append(newfilename)

                occlus.append(self.occlus[i])
                # newBoundingBoxes.append(tempBbox)
                # count += 1

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)
        self.occlus = np.array(occlus)
        self.newFilenames = np.array(newFilenames)

        self.newOccluImgs = np.array(newOccluImgs)
        self.newOcclus = np.array(newOcclus)
        self.newOccluInitLandmarks = np.array(newOccluInitLandmarks)
        self.newOccluGtLandmarks = np.array(newOccluGtLandmarks)
        # print(self.newOccluGtLandmarks)
        self.newOccluFilenames = np.array(newOccluFilenames)
        
        # self.boundingBoxes = np.array(newBoundingBoxes)
        # print(self.occlus.shape)
        # print(self.newOcclus)
        # print(self.newOcclus.shape)

        # print(self.occluImgs.shape)

    def CropResizeRotateAll(self):
        newImgs = []  
        newGtLandmarks = []
        newInitLandmarks = []   
        # output_dir = './tmp' 
        # if not os.path.exists(output_dir):
        #     os.mkdir(output_dir)
        for i in range(self.initLandmarks.shape[0]):
            # print(self.filenames[i])
            img = self.imgs[i]
            initLandmark = self.initLandmarks[i]
            gtLandmark = self.gtLandmarks[i]
            # cv2.imwrite("../data/roughFaceAlignment/{}.png".format(i), np.transpose(img, (1, 2, 0)))
            # np.savetxt("../data/roughFaceAlignment/{}.npz".format(i), initLandmark)
            # np.savetxt("../data/roughFaceAlignment/{}.pnpz".format(i), gtLandmark)
            tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(img, initLandmark, gtLandmark)
            # cv2.imwrite("../data/roughFaceAlignment/{}.png".format(i), np.transpose(tempImg, (1, 2, 0)))
            # np.savetxt("../data/roughFaceAlignment/{}.npz".format(i), tempInit)
            # np.savetxt("../data/roughFaceAlignment/{}.pnpz".format(i), tempGroundTruth)
            
            # prefix = os.path.splitext('_'.join(self.filenames[i].split('/')[3:]))[0]
            # tempImg =  tempImg.transpose((1, 2, 0))
            # print(tempImg)
            # print(filename_tmp) 
            # cv2.imwrite(os.path.join(output_dir, prefix+'.jpg'), tempImg)
            # np.savetxt(os.path.join(output_dir, prefix + '.initpts'), tempInit)
            # np.savetxt(os.path.join(output_dir, prefix + '.groundtpts'), tempGroundTruth)
            newImgs.append(tempImg)
            newInitLandmarks.append(tempInit)
            newGtLandmarks.append(tempGroundTruth)

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)  

    def NormalizeImages(self, imageServer=None):
        self.imgs = self.imgs.astype(np.float32)

        if imageServer is None:
            self.meanImg = np.mean(self.imgs, axis=0)
        else:
            self.meanImg = imageServer.meanImg

        self.imgs = self.imgs - self.meanImg
        
        if imageServer is None:
            self.stdDevImg = np.std(self.imgs, axis=0)
        else:
            self.stdDevImg = imageServer.stdDevImg
        
        self.imgs = self.imgs / self.stdDevImg

        from matplotlib import pyplot as plt  

        meanImg = self.meanImg - self.meanImg.min()
        meanImg = 255 * meanImg / meanImg.max()  
        meanImg = meanImg.astype(np.uint8)   
        if self.color:
            plt.imshow(np.transpose(meanImg, (1, 2, 0)))
        else:
            plt.imshow(meanImg[0], cmap=plt.cm.gray)
        plt.savefig(self.datasetDir + "meanImg.png")
        plt.clf()

        stdDevImg = self.stdDevImg - self.stdDevImg.min()
        stdDevImg = 255 * stdDevImg / stdDevImg.max()  
        stdDevImg = stdDevImg.astype(np.uint8)   
        if self.color:
            plt.imshow(np.transpose(stdDevImg, (1, 2, 0)))
        else:
            plt.imshow(stdDevImg[0], cmap=plt.cm.gray)
        plt.savefig(self.datasetDir + "stdDevImg.png")
        plt.clf()

    def CropResizeRotate(self, img, initShape, groundTruth):
        # print(initShape)
        # print(groundTruth)
        # print('-----------')
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)

        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        destShape = scaledMeanShape.copy() - scaledMeanShape.mean(axis=0)
        offset = np.array(self.imgSize[::-1]) / 2
        destShape += offset

        A, t = utils.bestFit(destShape, initShape, True)
    
        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        outImg = np.zeros((img.shape[0], self.imgSize[0], self.imgSize[1]), dtype=img.dtype)
        for i in range(img.shape[0]):
            outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=self.imgSize)

        initShape = np.dot(initShape, A) + t
        groundTruth = np.dot(groundTruth, A) + t
        # print(initShape)
        # print(groundTruth)
        return outImg, initShape, groundTruth

    def baselineShow(self):
        meanOcularDistance = 0.0
        meanocularSquareError = 0.0
        meanDeltaSquareError = 0.0
        occluNums = 0.0
        dataSize = len(self.imgs)
        meanOccluWeightedDeltaSquareError = 0.0

        for elem in self.roughLandmarks:
            gtLandmarks = np.reshape(elem[:136], (68, 2))
            firstStageLandmarks = np.reshape(elem[136:272], (68, 2))
            occlu = elem[272:340]
            landmarkError = elem[340:408]

            OcularDistance = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
            meanOcularDistance += OcularDistance
            # print(OccluDistance)

            ocularSquareError = np.mean(landmarkError)
            meanocularSquareError += ocularSquareError
            
            deltaSquareError = np.mean(np.sqrt(np.sum((gtLandmarks - firstStageLandmarks)**2,axis=1)))
            meanDeltaSquareError += deltaSquareError
            
            occluNums += np.sum(occlu)

            scaledOcclu = np.reshape(1 - 0.1 * occlu, (68, 1))    
            occluWeightedDeltaSquareError = np.mean(np.sqrt(np.sum((scaledOcclu * (gtLandmarks - firstStageLandmarks))**2,axis=1))) 
            meanOccluWeightedDeltaSquareError += occluWeightedDeltaSquareError

        meanOcularDistance /= dataSize
        meanocularSquareError /= dataSize
        meanDeltaSquareError /= dataSize
        occluRatio = occluNums / (68 * dataSize) 
        meanOccluWeightedDeltaSquareError /= dataSize
        
        print("dataSize: {}".format(dataSize))
        print("meanOcularDistance: {}".format(meanOcularDistance))
        print("meanocularSquareError: {}".format(meanocularSquareError))
        print("meanDeltaSquareError: {}".format(meanDeltaSquareError))
        print("occluRatio: {}".format(occluRatio))
        print("meanOccluWeightedDeltaSquareError: {}".format(meanOccluWeightedDeltaSquareError))
