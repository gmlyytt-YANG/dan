import cPickle as pickle
import pandas as pd
import os
import numpy as np
from ImageServer import ImageServer

class OccluServer(object):
    def __init__(self, DANmodel=None, imageServer=None):
        self.features = []
        self.labels = []
        self.DANmodel = DANmodel
        self.imageServer = imageServer

    def featureGenerating(self, trainLoad=False, normalized=False):
        nImgs = len(self.imageServer.imgs)
        
        for i in range(nImgs):
            img = self.imageServer.imgs[i]
            initLandmarks = self.imageServer.initLandmarks[i]
            gtLandmarks = self.imageServer.gtLandmarks[i]

            if trainLoad:
                prefix = os.path.splitext(self.imageServer.newFilenames[i].split('/')[-1])[0]
            else:
                prefix = os.path.splitext(self.imageServer.filenames[i].split('/')[-1])[0]
            # print(prefix)

            if img.shape[0] > 1:
                img = np.mean(img, axis=0)[np.newaxis]

            resLandmarks = initLandmarks
            fc1output = self.DANmodel.getFc1(img, resLandmarks, normalized=normalized)
            # print(fc1output.shape)
            self.features.append(fc1output)
        
        self.features = np.array(self.features)
        self.labels = self.imageServer.occlus
        print(self.features.shape, self.labels.shape)

    def Save(self, datasetDir, filename=None):
        arrays = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key) and key != 'DANmodel' and key != 'imageServer'}
        # np.savez(datasetDir + filename, **arrays)
        with open(datasetDir + filename, 'wb') as f:
            pickle.dump(arrays, f)

    @staticmethod
    def Load(filename):
        occluServer = OccluServer()
        with open(filename, 'rb') as f:
            arrays = pickle.load(f)
            occluServer.__dict__.update(arrays)
        
        return occluServer
