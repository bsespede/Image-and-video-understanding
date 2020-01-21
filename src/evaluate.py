from stvi_class.datapreprocessing import DataProcessor
from stvi_class.featureextraction import FeatureExtractor
import cv2
import json

def findVideo(videos, video_name):
    return next((item for item in video_files if item["video_name"] == video_name), None)

dps = DataProcessor('../data/processed_kept/keep/', '../data/Diving48_rgb/rgb')
fet = FeatureExtractor(dps)
svm = cv2.ml.SVM_load("./pose_classifier.svm")

video_name = '26__tigfCJFLZg_00124' # PIKE
video_name = '46_VNvb5oLOpLg_00145' # STR
video_name = '35_cYkUl8MrXgA_00010' # TUCK

with open("../data/processed_kept/labelling_results.json", "r") as read_file:
    video_files = json.load(read_file)

video = findVideo(video_files, video_name)
frame_range = (video['video_start'], video['video_end'])

dps.processVideo(video_name + '.pkl')
fet.processSTVIs(plotting=True, labelTrue=video['video_label'], classifier=svm, frameRange=frame_range)
