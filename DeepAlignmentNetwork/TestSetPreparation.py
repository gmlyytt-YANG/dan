from ImageServer import ImageServer
import numpy as np

imageBaseDir = "/media/kb250/K/yl/0_DATASET/file_bboxes_imgs_pts_opts/"
boundingBoxDir = "/media/kb250/K/yl/0_DATASET/file_bboxes_imgs_pts_opts/bboxes/"

commonSetImageDirs = [imageBaseDir + _ for _ in ["lfpw/testset/", "helen/testset/"]]
commonSetBoundingBoxFiles = [boundingBoxDir + _ for _ in  ["boxesLFPWTest.pkl", "boxesHelenTest.pkl"]]

challengingSetImageDirs = [imageBaseDir + "ibug/"]
challengingSetBoundingBoxFiles = [boundingBoxDir + "boxesIBUG.pkl"]

w300SetImageDirs = [imageBaseDir + _ for _ in ["300W/01_Indoor/", "300W/02_Outdoor/"]]
w300SetBoundingBoxFiles = [boundingBoxDir + _ for _ in ["boxes300WIndoor.pkl", "boxes300WOutdoor.pkl"]]

datasetDir = "/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/data/"

meanShape = np.load("/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/data/meanFaceShape.npz")["meanShape"]

commonSet = ImageServer(initialization='box')
commonSet.PrepareData(commonSetImageDirs, commonSetBoundingBoxFiles, meanShape, 0, 1000, False)
commonSet.LoadImages()
commonSet.CropResizeRotateAll()
commonSet.imgs = commonSet.imgs.astype(np.float32)
commonSet.Save(datasetDir, "commonSet0.npz")

challengingSet = ImageServer(initialization='box')
challengingSet.PrepareData(challengingSetImageDirs, challengingSetBoundingBoxFiles, meanShape, 0, 1000, False)
challengingSet.LoadImages()
challengingSet.CropResizeRotateAll()
challengingSet.imgs = challengingSet.imgs.astype(np.float32)
challengingSet.Save(datasetDir, "challengingSet0.npz")

w300Set = ImageServer(initialization='box')
w300Set.PrepareData(w300SetImageDirs, w300SetBoundingBoxFiles, meanShape, 0, 1000, False)
w300Set.LoadImages()
w300Set.CropResizeRotateAll()
w300Set.imgs = w300Set.imgs.astype(np.float32)
w300Set.Save(datasetDir, "w300Set0.npz")
