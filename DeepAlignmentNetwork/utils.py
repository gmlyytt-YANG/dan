import numpy as np
import time
import logging
import cv2

def scale(data):
    """Scale data to [0, 255] using min max method. """
    return np.multiply((data - np.min(data)) / (np.max(data) - np.min(data)), 255)

def scaleImgLandmark(img, box, landmark):
    if box is None:
        box = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])
    box = [int(_) for _ in box]
    w, h = box[2] - box[0], box[3] - box[1]
    scale = int(min(w, h) * 0.2)
    boxOri = np.array(box).copy()
    
    if box[1] >= scale:
        box[1] -= scale
    if box[3] <= img.shape[0] - scale:
        box[3] += scale
    if box[0] >= scale:
        box[0] -= scale
    if box[2] <= img.shape[1] - scale:
        box[2] += scale
    
    w, h = box[2] - box[0], box[3] - box[1]
    # face = img[boxOri[1]:boxOri[3], boxOri[0]:boxOri[2]]
    outline = img[box[1]:box[3], box[0]:box[2]]
    landmark = np.multiply((landmark - [box[0], box[1]]) / [w, h], [outline.shape[1], outline.shape[0]])
    # print(outline.shape)
    # print(landmark)
    # print('---------')
    
    return outline, landmark, np.array([boxOri[0] - box[0], boxOri[1] - box[1], boxOri[2] - box[0], boxOri[3] - box[1]])

def gaussian_noise(img, channel_first=False):
	"""Add gaussian noise to images """
	noised_img = None
	if channel_first:
		channel, row, col = img.shape
		img = np.transpose(img, (1, 2, 0))
	else:
		row, col, channel = img.shape
	mean = 0
	var = 5
	sigma = var ** 0.5
	gauss = np.random.normal(mean, sigma, (row, col, channel))
	gauss = gauss.reshape(row, col, channel)
	noised_img = img + gauss
	noised_img = scale(noised_img)
	# show(noised_img)
	if channel_first:
		return np.transpose(noised_img, (2, 0, 1))
	return noised_img

def logger(msg):
    """Get logger format"""
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] - %(levelname)s: %(message)s')
    logging.info(msg)

def loadFromPts(filename):
    landmarks = np.genfromtxt(filename, skip_header=3, skip_footer=1)
    landmarks = landmarks - 1

    return landmarks

def loadFromOPts(filename):
    landmarks = np.genfromtxt(filename)
    landmarks[:, :-1] = landmarks[:, :-1] - 1

    return landmarks

def saveToPts(filename, landmarks):
    pts = landmarks + 1

    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(filename, pts, delimiter=' ', header=header, footer='}', fmt='%.3f', comments='')

def bestFitRect(points, meanS, box=None):
    if box is None:
        box = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
    boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ])

    boxWidth = box[2] - box[0]
    boxHeight = box[3] - box[1]

    meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
    meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

    scaleWidth = boxWidth / meanShapeWidth
    scaleHeight = boxHeight / meanShapeHeight
    scale = (scaleWidth + scaleHeight) / 2

    S0 = meanS * scale

    S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]    
    S0 += boxCenter - S0Center

    return S0

def bestFit(destination, source, returnTransform=False):
    destMean = np.mean(destination, axis=0)
    srcMean = np.mean(source, axis=0)

    srcVec = (source - srcMean).flatten()
    destVec = (destination - destMean).flatten()

    a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec)**2
    b = 0
    for i in range(destination.shape[0]):
        b += srcVec[2*i] * destVec[2*i+1] - srcVec[2*i+1] * destVec[2*i] 
    b = b / np.linalg.norm(srcVec)**2
    
    T = np.array([[a, b], [-b, a]])
    srcMean = np.dot(srcMean, T)

    if returnTransform:
        return T, destMean - srcMean
    else:
        return np.dot(srcVec.reshape((-1, 2)), T) + destMean

def mirrorShape(shape, imgShape=None):
    imgShapeTemp = np.array(imgShape)
    shape2 = mirrorShapes(shape.reshape((1, -1, 3)), imgShapeTemp.reshape((1, -1)))[0]

    return shape2

def mirrorShapes(shapes, imgShapes=None):
    shapes2 = shapes.copy()
    
    for i in range(shapes.shape[0]):
        if imgShapes is None:
            shapes2[i, :, 0] = -shapes2[i, :, 0]
        else:
            shapes2[i, :, 0] = -shapes2[i, :, 0] + imgShapes[i][1]
        
        lEyeIndU = range(36, 40)
        lEyeIndD = [40, 41]
        rEyeIndU = range(42, 46)
        rEyeIndD = [46, 47]
        lBrowInd = range(17, 22)
        rBrowInd = range(22, 27)
        
        uMouthInd = range(48, 55)
        dMouthInd = range(55, 60)
        uInnMouthInd = range(60, 65)
        dInnMouthInd = range(65, 68)
        noseInd = range(31, 36)
        beardInd = range(17)
         
        lEyeU = shapes2[i, lEyeIndU].copy()
        lEyeD = shapes2[i, lEyeIndD].copy()
        rEyeU = shapes2[i, rEyeIndU].copy()       
        rEyeD = shapes2[i, rEyeIndD].copy() 
        lBrow = shapes2[i, lBrowInd].copy()
        rBrow = shapes2[i, rBrowInd].copy()

        uMouth = shapes2[i, uMouthInd].copy()
        dMouth = shapes2[i, dMouthInd].copy()
        uInnMouth = shapes2[i, uInnMouthInd].copy()
        dInnMouth = shapes2[i, dInnMouthInd].copy()
        nose = shapes2[i, noseInd].copy()
        beard = shapes2[i, beardInd].copy()
        
        lEyeIndU.reverse()
        lEyeIndD.reverse()
        rEyeIndU.reverse()
        rEyeIndD.reverse()
        lBrowInd.reverse()
        rBrowInd.reverse()
        
        uMouthInd.reverse()
        dMouthInd.reverse()
        uInnMouthInd.reverse()
        dInnMouthInd.reverse()
        beardInd.reverse()
        noseInd.reverse()   
        
        shapes2[i, rEyeIndU] = lEyeU
        shapes2[i, rEyeIndD] = lEyeD
        shapes2[i, lEyeIndU] = rEyeU
        shapes2[i, lEyeIndD] = rEyeD
        shapes2[i, rBrowInd] = lBrow
        shapes2[i, lBrowInd] = rBrow
        
        shapes2[i, uMouthInd] = uMouth
        shapes2[i, dMouthInd] = dMouth
        shapes2[i, uInnMouthInd] = uInnMouth
        shapes2[i, dInnMouthInd] = dInnMouth
        shapes2[i, noseInd] = nose
        shapes2[i, beardInd] = beard
        
    return shapes2

