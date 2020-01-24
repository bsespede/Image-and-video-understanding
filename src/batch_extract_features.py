from stvi_class.datapreprocessing import DataProcessor
from stvi_class.featureextraction import FeatureExtractor
import json
import numpy as np
# import sys

dps = DataProcessor('../data/processed_kept/keep/')
fet = FeatureExtractor(dps)

with open("../data/processed_kept/labelling_results.json", "r") as read_file:
    video_files = json.load(read_file)

# video_files = ('tuck0.pkl', 'tuck1.pkl', 'tuck3.pkl', 'notwist0.pkl', 'notwist1.pkl', 'notwist2.pkl', 'notwist3.pkl', 'twist0.pkl', 'twist1.pkl', 'twist2.pkl', 'twist3.pkl')
# video_files = ('notwist2.pkl',)

tokenset_pike = np.empty([0, fet.num_scalar_features_per_stvi])
tokenset_straight = np.empty([0, fet.num_scalar_features_per_stvi])
tokenset_tuck = np.empty([0, fet.num_scalar_features_per_stvi])

num_videos = len(video_files)

label_pike = 'PIKE'
label_straight = 'STR'
label_tuck = 'TUCK'

# for idx, video_name in enumerate(videofiles):
for idx, video_file in enumerate(video_files):
    video_name = video_file['video_name'] + '.pkl'
    label = video_file['video_label']
    start_frame = video_file['video_start']
    end_frame = video_file['video_end']
    print('Process video (', idx+1, '/', num_videos,  ')', video_name)
    dps.processVideo(video_name)
    fet.processSTVIs(verbose=False)

    cropped_frames = fet.feature_vectors[start_frame:end_frame]
    nonzero_frames = cropped_frames[~np.isnan(cropped_frames).any(axis=1)]

    if label == label_pike:
        tokenset_pike = np.append(tokenset_pike, np.atleast_2d(nonzero_frames), axis=0)
        video_ids_pike = np.append(video_ids_pike, idx * np.ones(nonzero_frames.shape[0]))
        video_labels[idx] = int(0)
    elif label == label_straight:
        tokenset_straight = np.append(tokenset_straight, np.atleast_2d(nonzero_frames), axis=0)
        video_ids_straight = np.append(video_ids_straight, idx * np.ones(nonzero_frames.shape[0]))
        video_labels[idx] = int(1)

    elif label == label_tuck:
        tokenset_tuck = np.append(tokenset_tuck, np.atleast_2d(nonzero_frames), axis=0)
        video_ids_tuck = np.append(video_ids_tuck, idx * np.ones(nonzero_frames.shape[0]))
        video_labels[idx] = int(2)


    # fet.exportFeatureVector('fts_' + video_name)
    # sys._debugmallocstats()


np.save('traindata_pike.npy', tokenset_pike)
np.save('traindata_straight.npy', tokenset_straight)
np.save('traindata_tuck.npy', tokenset_tuck)

np.save('video_ids_pike.npy', video_ids_pike)
np.save('video_ids_straight.npy', video_ids_straight)
np.save('video_ids_tuck.npy', video_ids_tuck)

np.save('video_labels.npy', video_labels)
