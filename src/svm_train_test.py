import os
import numpy as np
import cv2 as cv
from scipy.signal import medfilt

def trainAndTest(featurePath='.', randomizeData=True, filterFeatures=True):
    scriptDirectory = os.path.dirname(__file__)
    pikeData = np.load(featurePath + '/traindata_pike.npy')
    straightData = np.load(featurePath + '/traindata_straight.npy')
    tuckData = np.load(featurePath + '/traindata_tuck.npy')
    allData = np.vstack((pikeData, straightData, tuckData))

    video_ids_pike = np.load(featurePath + '/video_ids_pike.npy')
    video_ids_straight =  np.load(featurePath + '/video_ids_straight.npy')
    video_ids_tuck = np.load(featurePath + '/video_ids_tuck.npy')
    video_ids = np.hstack((video_ids_pike, video_ids_straight, video_ids_tuck))

    video_labels = np.load(featurePath + '/video_labels.npy')

    # filter in feature space
    if filterFeatures:
        filter_kernel_size = 9
        for video_id in np.unique(video_ids):
            video_frame_idxs = np.argwhere(video_ids.astype(int) == video_id.astype(int)).flatten()
            video_frame_features = allData[video_frame_idxs,:]
            video_frame_features_filtered = np.zeros_like(video_frame_features.transpose())
            for idx, video_frame_feature in enumerate(video_frame_features.transpose()):
                video_frame_features_filtered[idx,:] = medfilt(video_frame_feature, filter_kernel_size)
            allData[video_frame_idxs,:] = video_frame_features_filtered.transpose()

        [pikeData, straightData, tuckData] = np.split(allData, [len(video_ids_pike), len(video_ids_pike)+len(video_ids_straight)])

    # randomly shuffle data
    if randomizeData:
        idxs_pike = np.random.permutation(pikeData.shape[0])
        pikeData = pikeData[idxs_pike]
        video_ids_pike = video_ids_pike[idxs_pike]

        idxs_straight = np.random.permutation(straightData.shape[0])
        straightData = straightData[idxs_straight]
        video_ids_straight = video_ids_straight[idxs_straight]

        idxs_tuck = np.random.permutation(tuckData.shape[0])
        tuckData = tuckData[idxs_tuck]
        video_ids_tuck = video_ids_tuck[idxs_tuck]

    # general stuff
    trainPercentage = 0.9
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

    # randomize training data
    # idxs = np.random.permutation(trainingData.shape[0])
    # trainingData = trainingData[idxs]
    # trainingLabels = trainingLabels.transpose()[idxs]
    # video_ids = video_ids[idxs]

    # train svm
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_NU_SVC)
    svm.setKernel(cv.ml.SVM_RBF) # TODO: try other interpolation methods
    svm.setNu(0.1)
    svm.setTermCriteria((cv.TERM_CRITERIA_EPS, int(5e3), 1e-3))
    svm.trainAuto(trainingData, cv.ml.ROW_SAMPLE, trainingLabels)

    # train accuracy
    print("Framewise classification:")
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
    # np.save('confusion_matrix_train.npy', confusionMatrixTrain)

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
    # np.save('confusion_matrix.npy', confusionMatrix)

    classificationAccuracy = np.diagonal(confusionMatrix) / np.sum(confusionMatrix, axis=1)
    print('Classification accuracy: ', classificationAccuracy)


    # videowise maximum voting
    print("\nVideowise classification:")

    training_video_labels = video_labels[np.unique(training_video_ids).astype(int)]
    training_video_response = np.zeros_like(np.unique(training_video_ids))
    for idx, video_id in enumerate(np.unique(training_video_ids)):
        video_frame_idxs = np.argwhere(training_video_ids.astype(int) == video_id.astype(int))
        training_video_frame_response = trainingResponse[video_frame_idxs]
        training_video_response[idx] = np.argmax(np.bincount(training_video_frame_response.flatten()))

    videoClassificationAccuracyTrain = np.sum(training_video_labels == training_video_response.astype(int)) / len(training_video_response)

    print('Video classification accuracy (train): ', videoClassificationAccuracyTrain)

    test_video_labels = video_labels[np.unique(test_video_ids).astype(int)]
    test_video_response = np.zeros_like(np.unique(test_video_ids))
    for idx, video_id in enumerate(np.unique(test_video_ids)):
        video_frame_idxs = np.argwhere(test_video_ids.astype(int) == video_id.astype(int))
        test_video_frame_response = testResponse[video_frame_idxs]
        test_video_response[idx] = np.argmax(np.bincount(test_video_frame_response.flatten()))

    videoClassificationAccuracy = np.sum(test_video_labels == test_video_response.astype(int)) / len(test_video_response)

    print('Video classification accuracy: ', videoClassificationAccuracy)

    # svm.save('pose_classifier.svm')

    return classificationAccuracyTrain, classificationAccuracy,\
        confusionMatrixTrain, confusionMatrix,\
        videoClassificationAccuracyTrain, videoClassificationAccuracy


if __name__ == '__main__':
    numRuns = 30
    featurePath = './'

    batch_classificationAccuracyTrain = np.zeros((numRuns, 3))
    batch_classificationAccuracy = np.zeros((numRuns, 3))
    batch_confusionMatrixTrain = np.zeros((numRuns, 3, 3), dtype=np.int64)
    batch_confusionMatrix = np.zeros((numRuns, 3, 3), dtype=np.int64)
    batch_videoClassificationAccuracyTrain = np.zeros((numRuns))
    batch_videoClassificationAccuracy = np.zeros((numRuns))

    for runNr in range(numRuns):
        print('======== Run', runNr, '========')
        classificationAccuracyTrain, classificationAccuracy,\
        confusionMatrixTrain, confusionMatrix,\
        videoClassificationAccuracyTrain, videoClassificationAccuracy\
        = trainAndTest(featurePath)
        batch_classificationAccuracyTrain[runNr,:] = classificationAccuracyTrain
        batch_classificationAccuracy[runNr,:] = classificationAccuracy
        batch_confusionMatrixTrain[runNr,:,:] = confusionMatrixTrain
        batch_confusionMatrix[runNr,:,:] = confusionMatrix
        batch_videoClassificationAccuracyTrain[runNr] = videoClassificationAccuracyTrain
        batch_videoClassificationAccuracy[runNr] = videoClassificationAccuracy

    mean_classificationAccuracyTrain = np.mean(batch_classificationAccuracyTrain, axis=0)
    mean_classificationAccuracy = np.mean(batch_classificationAccuracy, axis=0)
    mean_confusionMatrixTrain = np.mean(batch_confusionMatrixTrain, axis=0)
    mean_confusionMatrix = np.mean(batch_confusionMatrix, axis=0)
    mean_videoClassificationAccuracyTrain = np.mean(batch_videoClassificationAccuracyTrain, axis=0)
    mean_videoClassificationAccuracy = np.mean(batch_videoClassificationAccuracy, axis=0)

    std_classificationAccuracyTrain = np.std(batch_classificationAccuracyTrain, axis=0)
    std_classificationAccuracy = np.std(batch_classificationAccuracy, axis=0)
    std_confusionMatrixTrain = np.std(batch_confusionMatrixTrain, axis=0)
    std_confusionMatrix = np.std(batch_confusionMatrix, axis=0)
    std_videoClassificationAccuracyTrain = np.std(batch_videoClassificationAccuracyTrain, axis=0)
    std_videoClassificationAccuracy = np.std(batch_videoClassificationAccuracy, axis=0)

    min_classificationAccuracyTrain = np.min(batch_classificationAccuracyTrain, axis=0)
    min_classificationAccuracy = np.min(batch_classificationAccuracy, axis=0)
    min_confusionMatrixTrain = np.min(batch_confusionMatrixTrain, axis=0)
    min_confusionMatrix = np.min(batch_confusionMatrix, axis=0)
    min_videoClassificationAccuracyTrain = np.min(batch_videoClassificationAccuracyTrain, axis=0)
    min_videoClassificationAccuracy = np.min(batch_videoClassificationAccuracy, axis=0)

    max_classificationAccuracyTrain = np.max(batch_classificationAccuracyTrain, axis=0)
    max_classificationAccuracy = np.max(batch_classificationAccuracy, axis=0)
    max_confusionMatrixTrain = np.max(batch_confusionMatrixTrain, axis=0)
    max_confusionMatrix = np.max(batch_confusionMatrix, axis=0)
    max_videoClassificationAccuracyTrain = np.max(batch_videoClassificationAccuracyTrain, axis=0)
    max_videoClassificationAccuracy = np.max(batch_videoClassificationAccuracy, axis=0)

    print('======== Overall Runs', numRuns, '========')
    print('classificationAccuracyTrain:\n mean:', mean_classificationAccuracyTrain, '\t std:', std_classificationAccuracyTrain)
    print(' min:', min_classificationAccuracyTrain, '\t max:', max_classificationAccuracyTrain)
    print('classificationAccuracy:\n mean:', mean_classificationAccuracy, '\t std:', std_classificationAccuracy)
    print(' min:', min_classificationAccuracy, '\t max:', max_classificationAccuracy)
    print('confusionMatrixTrain:\n mean:\n', mean_confusionMatrixTrain, '\n std:\n', std_confusionMatrixTrain)
    print(' min:\n', min_confusionMatrixTrain, '\n max:\n', max_confusionMatrixTrain)
    print('confusionMatrix:\n mean:\n', mean_confusionMatrix, '\n std:\n', std_confusionMatrix)
    print(' min:\n', min_confusionMatrix, '\n max:\n', max_confusionMatrix)
    print('videoClassificationAccuracyTrain:\n mean:', mean_videoClassificationAccuracyTrain, '\t std:', std_videoClassificationAccuracyTrain)
    print(' min:', min_videoClassificationAccuracyTrain, '\t max:', max_videoClassificationAccuracyTrain)
    print('videoClassificationAccuracy:\n mean:', mean_videoClassificationAccuracy, '\t std:', std_videoClassificationAccuracy)
    print(' min:', min_videoClassificationAccuracy, '\t max:', max_videoClassificationAccuracy)
