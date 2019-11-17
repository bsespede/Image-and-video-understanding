import sys
import json
import os
import shutil
import cv2

from PySide2.QtCore import QUrl, Slot
from PySide2.QtWidgets import QApplication
from PySide2.QtQml import QQmlApplicationEngine


@Slot(float, float, float, float)
def setResult(actionFirstPercentage, actionSecondPercentage, videoFirstPercentage, videoSecondPercentage):
    global _results
    video = cv2.VideoCapture(_inputs[_curVideo]['video_path'])
    maxFrame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    _results.append({
        'video_path': _inputs[_curVideo]['video_path'],
        'video_name': _inputs[_curVideo]['video_name'],
        'video_style': _inputs[_curVideo]['video_style'],
        'video_label': _inputs[_curVideo]['video_label'],
        'min_action': int(actionFirstPercentage * maxFrame),
        'max_action': int(actionSecondPercentage * maxFrame),
        'min_video': int(videoFirstPercentage * maxFrame),
        'max_video': int(videoSecondPercentage * maxFrame)
    })
    _classifiedStyles[_inputs[_curVideo]['video_style']] += 1


@Slot()
def getNextVideo():
    global _curVideo, _results, _inputs, _window, _classifiedStyles
    _curVideo = _curVideo + 1
    while _curVideo < len(_inputs) and _classifiedStyles[_inputs[_curVideo]['video_style']] >= _maxVideosToClassifyPerStyle:
        _curVideo = _curVideo + 1
    if _curVideo < len(_inputs):
        _window.setNextVideo(_inputs[_curVideo]['video_path'], _inputs[_curVideo]['video_style'] + '')
    else:
        writeResults()
        sys.exit(0)


def readJson():
    global _labels, _classifiedStyles, _inputs
    directory = os.path.dirname(__file__)
    inputJsonPath = os.path.join(directory, "Diving48_vocab.json")
    with open(inputJsonPath) as jsonFile:
        jsonData = json.load(jsonFile)
        id = 0;
        for elem in jsonData:
            if elem[2] == "NoTwis" and elem[3] != "FREE":
                _labels[id] = elem[3]
            id = id + 1
    inputJsonPath = os.path.join(directory, "Diving48_train.json")
    with open(inputJsonPath) as jsonFile:
        jsonData = json.load(jsonFile)
        for elem in jsonData:
            if elem['label'] in _labels.keys():
                _inputs.append({
                    'video_path': os.path.join(directory, "Diving48_rgb/rgb/" + elem['vid_name'] + '.mp4'),
                    'video_name': elem['vid_name'],
                    'video_style': _labels[elem['label']],
                    'video_label': elem['label']
                })


def writeResults():
    global _results
    with open('results.json', 'w') as outfile:
        json.dump(_results, outfile, indent=4)
    for elem in _results:
        directory = os.path.dirname(__file__)
        parsedVideosPath = os.path.join(directory, "Diving48_selection")
        videoPath = os.path.join(parsedVideosPath, str(elem['video_label']) + '_' + elem['video_name'])
        framesPath = os.path.join(videoPath, "frames")
        flowPath = os.path.join(videoPath, "opticalflow")
        originalVideoFile = elem['video_path']
        resultVideoFile = os.path.join(videoPath, elem['video_name'] + ".mp4")
        os.makedirs(videoPath, exist_ok=True)
        os.makedirs(framesPath, exist_ok=True)
        os.makedirs(flowPath, exist_ok=True)
        shutil.copy2(originalVideoFile, resultVideoFile)
        videoToFrames(originalVideoFile, framesPath, elem['min_video'], elem['max_video'])


def videoToFrames(sourceVideoFile, outputFolder, startFrame, endFrame):
    video = cv2.VideoCapture(sourceVideoFile)
    currentFrame = startFrame
    video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    while currentFrame < endFrame:
        ret, frame = video.read()
        frameFile = os.path.join(outputFolder, str(currentFrame - startFrame).zfill(6) + ".jpg")
        cv2.imwrite(frameFile, frame)
        currentFrame += 1


if __name__ == '__main__':
    #_labels = {13:"STR", 12:"PIKE", 47:"TUCK"}
    _labels = {}
    _classifiedStyles = {"PIKE": 0, "STR": 0, "TUCK": 0}
    _maxVideosToClassifyPerStyle = 3
    _curVideo = 0
    _results = []
    _inputs = []
    readJson()

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile('gui.qml'))

    _window = engine.rootObjects()[0]
    _window.setResult.connect(setResult)
    _window.getNextVideo.connect(getNextVideo)
    _window.setNextVideo(_inputs[_curVideo]['video_path'], _inputs[_curVideo]['video_style'])

    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec_())
