import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Layouts 1.13
import QtMultimedia 5.13

Rectangle {
    id: mainWindow
    width: 660
    height: 680
    color: "white"

    Component.onCompleted: videoPlayer.pause()

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
        }
    }
}
