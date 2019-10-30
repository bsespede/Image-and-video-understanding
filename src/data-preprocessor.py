import json
import os
import shutil
import cv2

def process_video(vid_name, start_frame, end_frame):
    print('preparing video ' + vid_name + ", start: " + str(start_frame) + ", end: " + str(end_frame))
    video_path = os.path.join(parsed_videos_path, vid_name)
    frames_path = os.path.join(video_path, "frames")
    flow_path = os.path.join(video_path, "opticalflow")
    original_video_file = os.path.join(videos_path, vid_name + ".mp4")
    video_file = os.path.join(video_path, vid_name + ".mp4")
    # make file structure
    os.mkdir(video_path)
    os.mkdir(frames_path)
    os.mkdir(flow_path)
    shutil.copy2(original_video_file, video_file)
    # store frames from video
    current_frame = start_frame
    video = cv2.VideoCapture(video_file)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while current_frame < end_frame:
        ret, frame = video.read()
        frame_file = str(current_frame - start_frame).zfill(6) + ".jpg"
        frame_path = os.path.join(frames_path, frame_file)
        cv2.imwrite(frame_path, frame)
        current_frame += 1


def process_jsons():
    # get the list of classes
    with open(vocab_json_path) as vocab_json:
        vocab_data = json.load(vocab_json)
        vocab_id = 0
        for elem in vocab_data:
            if 'Dive' in elem and 'NoTwis' in elem:
                labels.append(vocab_id)
            vocab_id += 1
    # get the list of videos of our class
    with open(training_json_path) as vocab_json:
        vocab_data = json.load(vocab_json)
        total_videos = 0
        for elem in vocab_data:
            if elem['label'] in labels:
                vid_name = elem['vid_name']
                start_frame = int(elem['start_frame'])
                end_frame = int(elem['end_frame'])
                process_video(vid_name, start_frame, end_frame)
            total_videos += 1


labels = []
directory = os.path.dirname(__file__)
vocab_json_path = os.path.join(directory, "Diving48_vocab.json")
training_json_path = os.path.join(directory, "Diving48_train.json")
videos_path = os.path.join(directory, "videos-diving")
parsed_videos_path = os.path.join(directory, "parsed-videos")
os.mkdir(parsed_videos_path)
process_jsons()
