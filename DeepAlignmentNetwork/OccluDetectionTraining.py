from __future__ import print_function
import os
import time
import datetime
import lasagne
from lasagne.layers import Conv2DLayer, batch_norm
from lasagne.nonlinearities import rectify, sigmoid
from lasagne.init import GlorotUniform

from matplotlib import pyplot as plt
import numpy as np
import theano
import theano.tensor as T
 
import utils

class OccluDetectionTraining(object):
    def __init__(self):
        self.batchsize = 64
        
        self.data = theano.tensor.tensor4('inputs', dtype=theano.config.floatX)
        self.targets = theano.tensor.tensor4('targets')

        self.occluRecalls = []
        self.clearRecalls = []

    def loadData(self, trainSet, validationSet):
        self.nSamples = trainSet.imgs.shape[0] + trainSet.newOccluImgs.shape[0]
        # print(trainSet.imgs.shape)
        # print(trainSet.newOccluImgs.shape)
        # print(self.nSamples)
        self.imageHeight = trainSet.imgSize[0]
        self.imageWidth = trainSet.imgSize[1]
        self.nChannels = trainSet.imgs.shape[1]

        self.Xtrain = np.vstack((trainSet.imgs, trainSet.newOccluImgs)).astype(np.float32) 
        # print(self.Xtrain.shape)

        self.Xvalid = validationSet.imgs.astype(np.float32)

        self.Ytrain = np.zeros((self.nSamples, 1, 1, 68), dtype=np.int8)
        occlus = np.expand_dims(np.vstack((trainSet.occlus, trainSet.newOcclus)), axis=1)
        # print(occlus.shape)
        self.Ytrain[:, 0] = occlus 

        self.Yvalid = np.zeros((validationSet.imgs.shape[0], 1, 1, 68), dtype=np.int8) 
        occlus = np.expand_dims(validationSet.occlus, axis=1)
        # print(occlus.shape)
        self.Yvalid[:, 0] = occlus

        self.testIdxsTrainSet = range(len(self.Xvalid))
        self.testIdxsValidSet = range(len(self.Xvalid))
        
        self.meanImg = trainSet.meanImg
        self.stdDevImg = trainSet.stdDevImg
        self.initLandmarks = trainSet.initLandmarks[0]

    def createCNN(self):
        net = {}
        net['input'] = lasagne.layers.InputLayer(shape=(None, self.nChannels, self.imageHeight, self.imageWidth), input_var=self.data)
        print("Input shape: {0}".format(net['input'].output_shape))
        
        #STAGE 1
        net['conv1_1'] = batch_norm(Conv2DLayer(net['input'], 64, 3, pad='same', W=GlorotUniform('relu')))
        net['conv1_2'] = batch_norm(Conv2DLayer(net['conv1_1'], 64, 3, pad='same', W=GlorotUniform('relu')))
        net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1_2'], 2)
        
        net['conv2_1'] = batch_norm(Conv2DLayer(net['pool1'], 128, 3, pad=1, W=GlorotUniform('relu')))
        net['conv2_2'] = batch_norm(Conv2DLayer(net['conv2_1'], 128, 3, pad=1, W=GlorotUniform('relu')))
        net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2_2'], 2)
        
        net['conv3_1'] = batch_norm (Conv2DLayer(net['pool2'], 256, 3, pad=1, W=GlorotUniform('relu')))
        net['conv3_2'] = batch_norm (Conv2DLayer(net['conv3_1'], 256, 3, pad=1, W=GlorotUniform('relu')))
        net['pool3'] = lasagne.layers.Pool2DLayer(net['conv3_2'], 2)
        
        # net['conv4_1'] = batch_norm(Conv2DLayer(net['pool3'], 512, 3, pad=1, W=GlorotUniform('relu')))
        # net['conv4_2'] = batch_norm (Conv2DLayer(net['conv4_1'], 512, 3, pad=1, W=GlorotUniform('relu')))
        # net['pool4'] = lasagne.layers.Pool2DLayer(net['conv4_2'], 2)
        
        net['fc1_dropout'] = lasagne.layers.DropoutLayer(net['pool3'], p=0.5)
        # net['s1_fc1'] = batch_norm(lasagne.layers.DenseLayer(net['s1_fc1_dropout'], num_units=256, nonlinearity=None))
        net['fc1'] = batch_norm(lasagne.layers.DenseLayer(net['fc1_dropout'], num_units=256, W=GlorotUniform('relu')))
        
        net['output'] = lasagne.layers.DenseLayer(net['fc1'], num_units=68, nonlinearity=None)

        return net

    def initializeNetwork(self):
        self.layers = self.createCNN()
        self.network = self.layers['output']

        self.prediction = lasagne.layers.get_output(self.network, deterministic=False)
        self.prediction_test = lasagne.layers.get_output(self.network, deterministic=True)

        self.generate_network_output = theano.function([self.data], [self.prediction])
        self.generate_network_output_deterministic = theano.function([self.data], [self.prediction_test])

        self.loss = self.weightedBinarySigmoid(self.prediction, self.targets)
        # self.test_loss = self.recallCompute(self.prediction_test, self.targets)

        # self.test_fn = theano.function([self.data, self.targets], self.test_loss)

    def weightedBinarySigmoid(self, predictions, targets):
        errors, updates = theano.scan(self.WBSCompute, [predictions, targets])
        
        return T.mean(errors)

    def WBSCompute(self, prediction, target):
        target = target[0]
        weight = 1.8
        # res = - (weight * target * T.log(prediction) + (1 - target) * T.log(1.0 - prediction))
        log_weight = 1 + (weight - 1) * target
        res = (1 - target) * prediction + log_weight * (T.log(1 + T.exp(-T.abs_(prediction))) + T.nnet.relu(-prediction))

        return T.mean(res)

    # def recallCompute(self, predictions, targets):
    #     recalls, updates = theano.scan(self.rCompute, [predictions, targets])

    #     # return [T.mean(recalls[0]), T.mean(recalls[1])]
    #     return T.mean(recalls)

    # def rCompute(self, prediction, target):
    #     target = target[0]
    #     threshold = 0.5
    #     # prediction[prediction >= threshold] = 1.0
    #     # prediction[prediction < threshold] = 0.0
    #     prediction = (T.sgn(prediction - threshold) + 1) / 2
    #     occluRecallRatio = prediction * target / T.sum(target)
    #     clearRecallRatio = (1 - prediction) * (1 - target) / T.sum(1 - target)
    #     
    #     # return [occluRecallRatio, clearRecallRatio]
    #     return occluRecallRatio

    # def getRecalls(self, X, y, loss, idxs, chunkSize=50):
    #     occluRecallList = []

    #     nImages = len(idxs)
    #     nChunks = 1 + nImages / chunkSize

    #     idxs = np.array_split(idxs, nChunks)
    #     
    #     print('stat occlu recall')
    #     for i in range(len(idxs)):
    #         occluRecall = loss(X[idxs[i]], y[idxs[i]])
    #         print(occluRecall)
    #         occluRecallList.append(occluRecall)
    #     print('----------')
    #     return np.mean(occluRecallList)

    def validateNetwork(self):
        threshold = 0.5
        outputList = []
        
        for index in range(len(self.Xvalid)):
            img = self.Xvalid[index]
            # label = self.Yvalid[index]
            output = self.generate_network_output_deterministic([img])[0][0]
            # print(output)
            output = (np.sign(output - threshold) + 1) / 2
            # print(output)
            # print('----------')
            outputList.append(output)
        
        targets = self.Yvalid.flatten()
        predictions = np.array(outputList).flatten()
        
        occluRecallRatio = np.dot(predictions, targets) / np.sum(targets)
        clearRecallRatio = np.dot(1 - predictions, 1 - targets) / np.sum(1 - targets)
        # print(targets)
        # print(predictions)
        print("occluRecall : {}, clearRecall: {}".format(occluRecallRatio, clearRecallRatio))
        # occluMeanRecall = self.getRecalls(self.Xvalid, self.Yvalid, self.test_fn, self.testIdxsValidSet)
         
        self.occluRecalls.append(occluRecallRatio)
        self.clearRecalls.append(clearRecallRatio)

        recallDir = "/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/network-occlu/network-{}".format(self.networkDes)
        if not os.path.exists(recallDir):
            os.makedirs(recallDir)
        np.savetxt(recallDir + "/" + "occluRecall.txt", self.occluRecalls)
        np.savetxt(recallDir + "/" + "clearRecall.txt", self.clearRecalls)

    def saveNetwork(self, dir="/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/networks/"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        network_filename =\
            dir + "network_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        np.savez(network_filename, *lasagne.layers.get_all_param_values(self.layers["output"]),
            occluRecalls=self.occluRecalls, clearRecalls=self.clearRecalls, meanImg = self.meanImg, stdDevImg=self.stdDevImg)

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def train(self, learning_rate = 0.05, num_epochs=10000, networkDes=""):
        self.networkDes = networkDes
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(self.loss, params, learning_rate=learning_rate)

        self.train_fn = theano.function([self.data, self.targets], self.loss, updates=updates)

        utils.logger("Starting training...")
        self.validateNetwork()

        highestOccluRecall = np.max(self.occluRecalls)
        highestClearRecall = np.max(self.clearRecalls)
        epochHighestOccluRecall = -1
        epochHighestClearRecall = -1 

        # print(highestOccluRecall)
        # print(highestClearRecall)

        earlyStoppintPatienceCount = 0
        earlyStoppingPatienceIters = 5

        for epoch in range(num_epochs):
            eachEpochHighestOccluRecall = -1
            eachEpochHighestClearRecall = -1

            utils.logger("Starting epoch " + str(epoch))

            train_err = 0
            train_batches = 0
            start_time = time.time()
            
            for batch in self.iterate_minibatches(self.Xtrain, self.Ytrain, self.batchsize, True): 
                inputs, targets = batch
                # print(inputs.shape)
                # print(targets.shape)
                train_batches += 1
                train_err_elem = self.train_fn(inputs, targets)
                # print(train_err_elem)
                train_err += train_err_elem

                if train_batches % 40 == 0:
                    self.validateNetwork()
                    if self.occluRecalls[-1] >= highestOccluRecall and self.clearRecalls[-1] >= highestClearRecall * 0.99:
                        save_dir = "/media/kb250/K/yl/10_DeepOccluAlignmentNetwork/network-occlu/network-{}/".format(networkDes)
                        if not os.path.exists(save_dir):
                            os.makesdir(save_dir)
                        self.saveNetwork(save_dir)
                        highestOccluRecall = self.occluRecalls[-1]
                        highestClearRecall = self.clearRecalls[-1]

                    if self.occluRecalls[-1] >= eachEpochHighestOccluRecall and self.clearRecalls[-1] >= eachEpochHighestClearRecall * 0.99:
                        eachEpochHighestOccluRecall = self.occluRecalls[-1]
                        eachEpochHighestClearRecall = self.clearRecalls[-1]
                
            print("training batch loss: {}".format(train_err / train_batches))

            if eachEpochHighestOccluRecall >= epochHighestOccluRecall and eachEpochHighestClearRecall >= epochHighestClearRecall * 0.99:
                epochHighestOccluRecall = eachEpochHighestOccluRecall
                epochHighestClearRecall = eachEpochHighestClearRecall
                earlyStoppintPatienceCount = 0
                print("new highest occluRecall: {}, new highest clearRecall: {}".format(highestOccluRecall, highestClearRecall))
            else:
                earlyStoppintPatienceCount += 1
                print("the highest occluRecall: {}, the highest clearRecall: {}".format(highestOccluRecall, highestClearRecall))
                print("this epoch occluRecall: {}, clearRecall: {}".format(eachEpochHighestOccluRecall, eachEpochHighestClearRecall))
                print("patience count: {}".format(earlyStoppintPatienceCount))

            if earlyStoppintPatienceCount >= earlyStoppingPatienceIters:
                break
