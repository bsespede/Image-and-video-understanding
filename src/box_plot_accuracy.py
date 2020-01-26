import matplotlib.pyplot as plt
import numpy as np

featurePath = './'

batch_classificationAccuracyTrain = np.load(featurePath + '/batch_classificationAccuracyTrain.npy')
batch_classificationAccuracy = np.load(featurePath + '/batch_classificationAccuracy.npy')
batch_confusionMatrixTrain = np.load(featurePath + '/batch_confusionMatrixTrain.npy')
batch_confusionMatrix = np.load(featurePath + '/batch_confusionMatrix.npy')
batch_videoClassificationAccuracyTrain = np.load(featurePath + '/batch_videoClassificationAccuracyTrain.npy')
batch_videoClassificationAccuracy = np.load(featurePath + '/batch_videoClassificationAccuracy.npy')

frame_class_fig, frame_class_axs = plt.subplots()
frame_class_axs.boxplot(batch_classificationAccuracyTrain, positions=[1, 4, 7], widths=0.35, notch=True)
frame_class_axs.boxplot(batch_classificationAccuracy, positions=[2, 5, 8], widths=0.35, notch=True)
frame_class_axs.set_title("Frame-wise classification accuracy")
frame_class_axs.set(xticks=[1, 2, 4, 5, 7, 8], xticklabels=['pike\ntrain', 'pike', 'str\ntrain', 'str', 'tuck\ntrain', 'tuck'])
frame_class_axs.set_xlim(0,9)

video_class_fig, video_class_axs = plt.subplots()
video_class_axs.boxplot(batch_videoClassificationAccuracyTrain, positions=[1, 4, 7], widths=0.35, notch=True)
video_class_axs.boxplot(batch_videoClassificationAccuracy, positions=[2, 5, 8], widths=0.35, notch=True)
video_class_axs.set_title("Video-wise classification accuracy")
video_class_axs.set(xticks=[1, 2, 4, 5, 7, 8], xticklabels=['pike\ntrain', 'pike', 'str\ntrain', 'str', 'tuck\ntrain', 'tuck'])
video_class_axs.set_xlim(0,9)

frame_class_fig.savefig(featurePath+'frame_class.png')
video_class_fig.savefig(featurePath+'video_class.png')

plt.show()
