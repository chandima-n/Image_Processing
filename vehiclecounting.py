import cv2
import dlib
import imutils
import numpy as np
from imutils.video import VideoStream, FPS
import time

# Assuming arguments, ClassNames, and trackingvehicle are defined elsewhere in your code

# Initialize the network
net = cv2.dnn.readNetFromCaffe(arguments["prototxt"], arguments["model"])

# Start video stream
if not arguments.get("input", False):
    print("Starting Video from Webcam...")
    source_video = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    print("Opening Video File...")
    source_video = cv2.VideoCapture(arguments["input"])

store = None
Width = None
Height = None
center = Cent_tracker(max_Disapp=40, max_Dist=50)
trackers = []
trackablevehicles = {}
T_Frames = 0
enter = 0
exit = 0
queue = 0
frames_p_second = FPS().start()

while True:
    frame = source_video.read()
    frame = frame[1] if arguments.get("input", False) else frame

    if arguments["input"] is not None and frame is None:
        break

    frame = imutils.resize(frame, width=300)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if Width is None or Height is None:
        (Height, Width) = frame.shape[:2]

    if arguments["output"] is not None and store is None:
        four_code_character = cv2.VideoWriter_fourcc(*"MJPG")
        store = cv2.VideoWriter(arguments["output"], four_code_character, 30,
                                (Width, Height), True)

    situation = "Sleep"
    boxes = []

    if T_Frames % arguments["skip_frames"] == 0:
        situation = "Observing"
        trackers = []

        net.setInput(cv2.dnn.blobFromImage(frame, 0.007843, (Width, Height), 127.5))
        output = net.forward()

        for i in np.arange(0, output.shape[2]):
            confidence = output[0, 0, i, 2]

            if confidence > arguments["confidence"]:
                classid = int(output[0, 0, i, 1])

                if ClassNames[classid] != "car":
                    continue

                box = output[0, 0, i, 3:7] * np.array([Width, Height, Width, Height])
                (box_X, box_Y, box_W, box_H) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(box_X), int(box_Y), int(box_W), int(box_H))
                tracker.start_track(rgb_frame, rect)
                trackers.append(tracker)

    else:
        for tracker in trackers:
            situation = "Tracking"
            tracker.update(rgb_frame)
            pos = tracker.get_position()

            box_X = int(pos.left())
            box_Y = int(pos.top())
            box_W = int(pos.right())
            box_H = int(pos.bottom())

            boxes.append((box_X, box_Y, box_W, box_H))

    cv2.line(frame, (0, Height // 2), (Width, Height // 2), (255, 0, 0), 2)
    vehicles = center.update(boxes)

    for (vehi_ID, centroid) in vehicles.items():
        t_vehi = trackablevehicles.get(vehi_ID, None)

        if t_vehi is None:
            t_vehi = trackingvehicle(vehi_ID, centroid)
        else:
            Y = [c[1] for c in t_vehi.centroids]
            headto = centroid[1] - np.mean(Y)
            t_vehi.centroids.append(centroid)

            if not t_vehi.counted:
                if headto < 0 and centroid[1] < Height:
                    exit += 1
                    t_vehi.counted = True
                elif headto > 0 and centroid[1] < Height:
                    enter += 1
                    t_vehi.counted = True

        queue = enter - exit
        trackablevehicles[vehi_ID] = t_vehi

        text = "ID {}".format(vehi_ID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    details = [
        ("Exit", exit),
        ("Enter", enter),
        ("Status", situation),
        ("Queue", queue),
    ]

    for (i, (p, q)) in enumerate(details):
        text = "{} - {}".format(p, q)
        cv2.putText(frame, text, (10, 100 - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    if store is not None:
        store.write(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

frames_p_second.stop()

if store is not None:
    store.release()

if not arguments.get("input", False):
    source_video.stop()
else:
    source_video.release()

cv2.destroyAllWindows()
