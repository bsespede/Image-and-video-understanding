import os
import numpy as np
import cv2 as cv
from scipy.signal import medfilt

if __name__ == '__main__':
    scriptDirectory = os.path.dirname(__file__)
    pikeData = np.load('hu_moments_0p5/traindata_pike.npy')
    straightData = np.load('hu_moments_0p5/traindata_straight.npy')
    tuckData = np.load('hu_moments_0p5/traindata_tuck.npy')
    allData = np.vstack((pikeData, straightData, tuckData))

    video_ids_pike = np.load('hu_moments_0p5/video_ids_pike.npy')
    video_ids_straight =  np.load('hu_moments_0p5/video_ids_straight.npy')
    video_ids_tuck = np.load('hu_moments_0p5/video_ids_tuck.npy')
    video_ids = np.hstack((video_ids_pike, video_ids_straight, video_ids_tuck))

    video_labels = np.load('hu_moments_0p5/video_labels.npy')

    # filter in feature space
    filter_kernel_size = 9
    for video_id in np.unique(video_ids):
        video_frame_idxs = np.argwhere(video_ids.astype(int) == video_id.astype(int)).flatten()
        video_frame_features = allData[video_frame_idxs,:]
        video_frame_features_filtered = np.zeros_like(video_frame_features.transpose())
        for idx, video_frame_feature in enumerate(video_frame_features.transpose()):
            video_frame_features_filtered[idx,:] = medfilt(video_frame_feature, filter_kernel_size)
        allData[video_frame_idxs,:] = video_frame_features_filtered.transpose()

    [pikeData, straightData, tuckData] = np.split(allData, [len(video_ids_pike), len(video_ids_pike)+len(video_ids_straight)])
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

    training_video_ids_pike = video_ids_pike[0:totalTrainingPikes]
    training_video_ids_straight = video_ids_straight[0:totalTrainingStraights]
    training_video_ids_tuck = video_ids_tuck[0:totalTrainingTucks]
    training_video_ids = np.concatenate((training_video_ids_pike, training_video_ids_straight, training_video_ids_tuck))

    trainingData = np.copy(pikeTrainingData)
    trainingData = np.append(trainingData, straightTrainingData, axis=0)
    trainingData = np.append(trainingData, tuckTrainingData, axis=0)

    # randomize
    # idxs = np.random.permutation(trainingData.shape[0])
    # trainingData = trainingData[idxs]
    # trainingLabels = trainingLabels.transpose()[idxs]
    # video_ids = video_ids[idxs]

    # train svm
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_NU_SVC)
    svm.setKernel(cv.ml.SVM_RBF) # TODO: try other interpolation methods
    svm.setNu(0.1)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e3), 1e-3))
    svm.trainAuto(trainingData, cv.ml.ROW_SAMPLE, trainingLabels)

    # train accuracy
    confusionMatrixTrain = np.zeros((3, 3), dtype=np.int64)

    trainingPikeResponse = np.zeros_like(pikeTrainingLabels)
    for idx, pikeFeatureVector in enumerate(pikeTrainingData):
        response = svm.predict(pikeFeatureVector.reshape((1, pikeFeatures)))[1]
        convertedResponse = int(response[0][0])
        trainingPikeResponse[0,idx] = convertedResponse
        confusionMatrixTrain[0, convertedResponse] += 1

    trainingStraightResponse = np.zeros_like(straightTrainingLabels)
    for idx, straightFeatureVector in enumerate(straightTrainingData):
        response = svm.predict(straightFeatureVector.reshape((1, straightFeatures)))[1]
        convertedResponse = int(response[0][0])
        trainingStraightResponse[0,idx] = convertedResponse
        confusionMatrixTrain[1, convertedResponse] += 1

    trainingTuckResponse = np.zeros_like(tuckTrainingLabels)
    for idx, tuckFeatureVector in enumerate(tuckTrainingData):
        response = svm.predict(tuckFeatureVector.reshape((1, tuckFeatures)))[1]
        convertedResponse = int(response[0][0])
        trainingTuckResponse[0,idx] = convertedResponse
        confusionMatrixTrain[2, convertedResponse] += 1

    trainingResponse = np.hstack((trainingPikeResponse, trainingStraightResponse, trainingTuckResponse)).transpose()

    print("Confusion matrix (train):\n", confusionMatrixTrain) # indices of conf matrix: 0 pike, 1 straight, 2 tuck
    np.save('confusion_matrix_train.npy', confusionMatrixTrain)

    classificationAccuracyTrain = np.diagonal(confusionMatrixTrain) / np.sum(confusionMatrixTrain, axis=1)
    print('Classification accuracy (train): ', classificationAccuracyTrain)

    # test related

    pikeTestData = pikeData[totalTrainingPikes:, :].astype(np.float32)
    straightTestData = straightData[totalTrainingStraights:, :].astype(np.float32)
    tuckTestData = tuckData[totalTrainingTucks:, :].astype(np.float32)

    totalTestPikes = pikeTestData.shape[0]
    totalTestStraights = straightTestData.shape[0]
    totalTestTucks = tuckTestData.shape[0]

    pikeTestLabels = np.full((1, totalTestPikes), 0, dtype=np.int64)
    straightTestLabels = np.full((1, totalTestStraights), 1, dtype=np.int64)
    tuckTestLabels = np.full((1, totalTestTucks), 2, dtype=np.int64)

    testLabels = np.concatenate((pikeTestLabels, straightTestLabels, tuckTestLabels), axis=1)

    test_video_ids_pike = video_ids_pike[totalTrainingPikes:]
    test_video_ids_straight = video_ids_straight[totalTrainingStraights:]
    test_video_ids_tuck = video_ids_tuck[totalTrainingTucks:]
    test_video_ids = np.concatenate((test_video_ids_pike, test_video_ids_straight, test_video_ids_tuck))

    confusionMatrix = np.zeros((3, 3), dtype=np.int64)

    # test svm_svm
    print("Framewise classification:")

    testPikeResponse = np.zeros_like(pikeTestLabels)
    for idx, pikeFeatureVector in enumerate(pikeTestData):
        response = svm.predict(pikeFeatureVector.reshape((1, pikeFeatures)))[1]
        convertedResponse = int(response[0][0])
        testPikeResponse[0,idx] = convertedResponse
        confusionMatrix[0, convertedResponse] += 1

    testStraightResponse = np.zeros_like(straightTestLabels)
    for idx, straightFeatureVector in enumerate(straightTestData):
        response = svm.predict(straightFeatureVector.reshape((1, straightFeatures)))[1]
        convertedResponse = int(response[0][0])
        testStraightResponse[0,idx] = convertedResponse
        confusionMatrix[1, convertedResponse] += 1

    testTuckResponse = np.zeros_like(tuckTestLabels)
    for idx, tuckFeatureVector in enumerate(tuckTestData):
        response = svm.predict(tuckFeatureVector.reshape((1, tuckFeatures)))[1]
        convertedResponse = int(response[0][0])
        testTuckResponse[0,idx] = convertedResponse
        confusionMatrix[2, convertedResponse] += 1

    testResponse = np.hstack((testPikeResponse, testStraightResponse, testTuckResponse)).transpose()

    print("Confusion matrix:\n", confusionMatrix) # indices of conf matrix: 0 pike, 1 straight, 2 tuck
    np.save('confusion_matrix.npy', confusionMatrix)

    classificationAccuracy = np.diagonal(confusionMatrix) / np.sum(confusionMatrix, axis=1)
    print('Classification accuracy: ', classificationAccuracy)


    # videowise maximum voting
    print("\nVideowise classification:")

    training_video_labels = video_labels[np.unique(training_video_ids).astype(int)]
    training_video_response = np.zeros_like(np.unique(training_video_ids))
    for idx, video_id in enumerate(np.unique(training_video_ids)):
        video_frame_idxs = np.argwhere(training_video_ids == video_id)
        training_video_frame_response = trainingResponse[video_frame_idxs]
        training_video_response[idx] = np.argmax(np.bincount(training_video_frame_response.flatten()))

    videoClassificationiAccuracyTrain = np.sum(training_video_labels == training_video_response.astype(int)) / len(training_video_response)

    print('Video classification accuracy (train): ', videoClassificationiAccuracyTrain)

    test_video_labels = video_labels[np.unique(test_video_ids).astype(int)]
    test_video_response = np.zeros_like(np.unique(test_video_ids))
    for idx, video_id in enumerate(np.unique(test_video_ids)):
        video_frame_idxs = np.argwhere(test_video_ids == video_id)
        test_video_frame_response = testResponse[video_frame_idxs]
        test_video_response[idx] = np.argmax(np.bincount(test_video_frame_response.flatten()))

    videoClassificationiAccuracy = np.sum(test_video_labels == test_video_response.astype(int)) / len(test_video_response)

    print('Video classification accuracy: ', videoClassificationiAccuracy)

    # svm.save('pose_classifier_0p5.svm')

