from ImageServer import ImageServer
import numpy as np

imageBaseDir = "../../0_DATASET/file_bboxes_imgs_pts_opts/"
boundingBoxDir = "../../0_DATASET/file_bboxes_imgs_pts_opts/bboxes/" 

imageDirs = [imageBaseDir + _ for _ in ["lfpw/trainset/", "helen/trainset/", "afw/"]]
boundingBoxFiles = [boundingBoxDir + _ for _ in ["boxesLFPWTrain.pkl", "boxesHelenTrain.pkl", "boxesAFW.pkl"]]

datasetDir = "../data/"

meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]

print('train')
trainSet = ImageServer(initialization='rect')
trainSet.PrepareData(imageDirs, None, meanShape, 100, 100000, True)
trainSet.LoadImages()
trainSet.GeneratePerturbations(10, [0.2, 0.2, 20, 0.25])
trainSet.NormalizeImages()
trainSet.Save(datasetDir)

print('val')
validationSet = ImageServer(initialization='box')
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 0, 100, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)
