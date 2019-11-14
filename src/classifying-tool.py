import sys
import json
import os

from PySide2.QtCore import QUrl, Slot
from PySide2.QtWidgets import QApplication
from PySide2.QtQml import QQmlApplicationEngine

@Slot(int, int, int, int)
def nextVideoClicked(actionFirst, actionSecond, videoFirst, videoSecond):
    global _curVideo, _results, _inputs, _window

    _results.append({
        'source_video': str(_inputs[_curVideo]['source_video']),
        'flight_style': str(_inputs[_curVideo]['flight_style']),
        'min_action': int(actionFirst),
        'max_action': int(actionSecond),
        'min_video': int(videoFirst),
        'max_video': int(videoSecond)
    })
    if _curVideo + 1 < len(_inputs):
        # read next video
        _curVideo += 1
        _window.loadNextVideo(_inputs[_curVideo]['source_video'], _inputs[_curVideo]['flight_style'])
    else:
        # dump results JSON
        directory = os.path.dirname(__file__)
        input_json_path = os.path.join(directory, "output_classifying_tool.json")
        with open(input_json_path, 'w') as output_file:
            json.dump(_results, output_file)
            _window = -1;


def readJSON():
    directory = os.path.dirname(__file__)
    input_json_path = os.path.join(directory, "input_classifying_tool.json")
    with open(input_json_path) as input_json:
        input_data = json.load(input_json)
        for elem in input_data:
            _inputs.append({
                'source_video': str(elem['source_video']),
                'flight_style': str(elem['flight_style'])
            })


if __name__ == '__main__':

    _window = 0
    _curVideo = 0
    _results = []
    _inputs = [{
        'source_video': 'video.mp4',
        'flight_style': 'Pike'
    }, {
        'source_video': 'video2.mp4',
        'flight_style': 'Tuck'
    }]
    # readJSON()

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile('gui.qml'))
    _window = engine.rootObjects()[0]
    _window.nextVideo.connect(nextVideoClicked)
    _window.loadNextVideo(_inputs[_curVideo]['source_video'], _inputs[_curVideo]['flight_style'])

    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec_())