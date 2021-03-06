from ImageServer import ImageServer
import numpy as np

dataBase = "../../0_DATASET/"
imageBaseDir = dataBase + "file_bboxes_imgs_pts_opts/"
boundingBoxDir = dataBase + "file_bboxes_imgs_pts_opts/bboxes/" 

imageDirs = [imageBaseDir + _ for _ in ["lfpw/trainset/", "helen/trainset/", "afw/", "lfpw/testset/", "helen/testset/", "ibug/", "300W/01_Indoor/", "300W/02_Outdoor/"]]
boundingBoxFiles = [boundingBoxDir + _ for _ in ["boxesLFPWTrain.pkl", "boxesHelenTrain.pkl", "boxesAFW.pkl", "boxesLFPWTest.pkl", "boxesHelenTest.pkl", "boxesIBUG.pkl", "boxes300WIndoor.pkl", "boxes300WOutdoor.pkl"]]

datasetDir = "../data/"

meanShape = np.load(datasetDir + "meanFaceShape.npz")["meanShape"]

print('train')
trainSet = ImageServer(initialization='rect', datasetDir=datasetDir)
trainSet.PrepareData(imageDirs, None, meanShape, 0, 3148, True)
trainSet.LoadImages()
trainSet.GeneratePerturbations(10, [0.2, 0.2, 10, 0.25])
trainSet.NormalizeImages()
trainSet.Save()

print('val')
validationSet = ImageServer(initialization='box', datasetDir=datasetDir)
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 3148, 1000000, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save()
