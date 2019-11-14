import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Layouts 1.13
import QtMultimedia 5.13

ApplicationWindow {
    id: root
    title: qsTr("[IVU] Video classification")
    width: 660
    height: 680
    color: "white"
    visible: true

    signal nextVideo(int actionFirst, int actionSecond, int videoFirst, int videoSecond)

    function loadNextVideo(videoName, styleName) {
        flightStyle.text = styleName;
        videoPlayer.source = videoName;
        videoPlayer.pause();
        videoTimer.restart();
    }

    Timer {
        id: videoTimer
        interval: 100
        running: false

        onTriggered: {
            actionSlider.from = 0;
            actionSlider.to = videoPlayer.duration;
            actionSlider.first.value = 0;
            actionSlider.second.value = videoPlayer.duration;

            videoSlider.from = 0;
            videoSlider.to = videoPlayer.duration;
            videoSlider.first.value = 0;
            videoSlider.second.value = videoPlayer.duration;
        }
    }

    ColumnLayout{
        anchors.margins:10
        anchors.fill: parent
        spacing: 10

        Text {
            id: flightStyle
            text: "Flight style"
            font.weight: Font.Bold
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
        }

        Video {
            id: videoPlayer
            width: 640
            height: 480
            source: "video.mp4"
            fillMode: VideoOutput.Stretch
            autoPlay: false
        }

        RowLayout {

            Text {
                text: "Action range:"
                Layout.minimumWidth: 100
                Layout.maximumWidth: 100
            }

            RangeSlider {
                id: actionSlider
                from: 0
                to: videoPlayer.duration
                first.value: 0
                second.value: actionSlider.to
                Layout.fillWidth: true

                first.onMoved: videoPlayer.seek(first.value)
                second.onMoved: videoPlayer.seek(second.value)
            }
        }

        RowLayout {

            Text {
                text: "Video range:"
                Layout.minimumWidth: 100
                Layout.maximumWidth: 100
            }

            RangeSlider {
                id: videoSlider
                from: 0
                to: videoPlayer.duration
                first.value: 0
                second.value: videoSlider.to
                Layout.fillWidth: true

                first.onMoved: videoPlayer.seek(first.value)
                second.onMoved: videoPlayer.seek(second.value)
            }
        }

        Button {
            id: nextBtn
            text: "Next video"
            Layout.fillWidth: true
            onClicked: root.nextVideo(actionSlider.first.value, actionSlider.second.value, videoSlider.first.value, videoSlider.second.value)
        }
    }
}
