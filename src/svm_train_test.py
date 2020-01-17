import os
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    scriptDirectory = os.path.dirname(__file__)
    pikeData = np.load('hu_moments/traindata_pike.npy')
    straightData = np.load('hu_moments/traindata_straight.npy')
    tuckData = np.load('hu_moments/traindata_tuck.npy')

    # general stuff
    trainPercentage = 0.9 #TODO: when increasing it always predicts  straights, data is unbalanced
    pikeFrames, pikeFeatures = pikeData.shape
    straightFrames, straightFeatures = straightData.shape
    tuckFrames, tuckFeatures = tuckData.shape

    # training related
    totalTrainingPikes = int(pikeFrames * trainPercentage)
    totalTrainingStraights = int(straightFrames * trainPercentage)
    totalTrainingTucks = int(tuckFrames * trainPercentage)

    pikeTrainingData = pikeData[:totalTrainingPikes, :].astype(np.float32)
    straightTrainingData = straightData[:totalTrainingStraights, :].astype(np.float32)
    tuckTrainingData = tuckData[:totalTrainingTucks, :].astype(np.float32)

    pikeTrainingLabels = np.full((1, totalTrainingPikes), 0, dtype=np.int64)
    straightTrainingLabels = np.full((1, totalTrainingStraights), 1, dtype=np.int64)
    tuckTrainingLabels = np.full((1, totalTrainingTucks), 2, dtype=np.int64)

    trainingLabels = np.concatenate((pikeTrainingLabels, straightTrainingLabels, tuckTrainingLabels), axis=1)

    trainingData = np.copy(pikeTrainingData)
    trainingData = np.append(trainingData, straightTrainingData, axis=0)
    trainingData = np.append(trainingData, tuckTrainingData, axis=0)

    # randomize
    idxs = np.random.permutation(trainingData.shape[0])
    trainingData = trainingData[idxs]
    trainingLabels = trainingLabels.transpose()[idxs]

    # train svm
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_NU_SVC)
    svm.setKernel(cv.ml.SVM_RBF) # TODO: try other interpolation methods
    svm.setNu(0.1)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e3), 1e-3))
    svm.trainAuto(trainingData, cv.ml.ROW_SAMPLE, trainingLabels)

    # train accuracy
    confusionMatrixTrain = np.zeros((3, 3), dtype=np.int64)

    for pikeFeatureVector in pikeTrainingData:
        response = svm.predict(pikeFeatureVector.reshape((1, pikeFeatures)))[1]
        convertedResponse = int(response[0][0])
        confusionMatrixTrain[0, convertedResponse] += 1

    for straightFeatureVector in straightTrainingData:
        response = svm.predict(straightFeatureVector.reshape((1, straightFeatures)))[1]
        convertedResponse = int(response[0][0])
        confusionMatrixTrain[1, convertedResponse] += 1

    for tuckFeatureVector in tuckTrainingData:
        response = svm.predict(tuckFeatureVector.reshape((1, tuckFeatures)))[1]
        convertedResponse = int(response[0][0])
        confusionMatrixTrain[2, convertedResponse] += 1

    print("Confusion matrix (train):\n", confusionMatrixTrain) # indices of conf matrix: 0 pike, 1 straight, 2 tuck
    np.save('confusion_matrix_train.npy', confusionMatrixTrain)

    classificationAccuracyTrain = np.diagonal(confusionMatrixTrain) / np.sum(confusionMatrixTrain, axis=1)
    print('Classification accuracy (train): ', classificationAccuracyTrain)

    # test related
    totalTestPikes = int(pikeFrames * trainPercentage)
    totalTestStraights = int(straightFrames * trainPercentage)
    totalTestTucks = int(tuckFrames * trainPercentage)

    pikeTestData = pikeData[totalTestPikes:, :].astype(np.float32)
    straightTestData = straightData[totalTestStraights:, :].astype(np.float32)
    tuckTestData = tuckData[totalTestTucks:, :].astype(np.float32)

    confusionMatrix = np.zeros((3, 3), dtype=np.int64)

    # test svm_svm
    for pikeFeatureVector in pikeTestData:
        response = svm.predict(pikeFeatureVector.reshape((1, pikeFeatures)))[1]
        convertedResponse = int(response[0][0])
        confusionMatrix[0, convertedResponse] += 1

    for straightFeatureVector in straightTestData:
        response = svm.predict(straightFeatureVector.reshape((1, straightFeatures)))[1]
        convertedResponse = int(response[0][0])
        confusionMatrix[1, convertedResponse] += 1

    for tuckFeatureVector in tuckTestData:
        response = svm.predict(tuckFeatureVector.reshape((1, tuckFeatures)))[1]
        convertedResponse = int(response[0][0])
        confusionMatrix[2, convertedResponse] += 1

    print("Confusion matrix:\n", confusionMatrix) # indices of conf matrix: 0 pike, 1 straight, 2 tuck
    np.save('confusion_matrix.npy', confusionMatrix)

    classificationAccuracy = np.diagonal(confusionMatrix) / np.sum(confusionMatrix, axis=1)
    print('Classification accuracy: ', classificationAccuracy)

    svm.save('pose_classifier.svm')

