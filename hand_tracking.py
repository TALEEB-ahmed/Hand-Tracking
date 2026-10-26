import cv2
import mediapipe as mp 
import time


BaseOptions = (mp.tasks.BaseOptions)
HandLandmarker = (mp.tasks.vision.HandLandmarker)
HandLandmarkerOptions = (mp.tasks.vision.HandLandmarkerOptions)
VisionRunningMode = (mp.tasks.vision.RunningMode)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

camera = cv2.VideoCapture(0)
# fps = camera.get(cv2.CAP_PROP_FPS)
# print(fps)
if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

with HandLandmarker.create_from_options(options) as detector:
    start_time = time.monotonic()
    previous_timestamp = -1

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)  

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int((time.monotonic() - start_time) * 1000)
        timestamp = max(timestamp, previous_timestamp + 1)
        previous_timestamp = timestamp

        results = detector.detect_for_video(mp_image, timestamp)

        if results.hand_landmarks:
            height, width, _ = frame.shape

            for landmarks in results.hand_landmarks:
                points = []

                for landmark in landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

                for connection in connections:
                    start = points[connection.start]
                    end = points[connection.end]
                    cv2.line(frame, start, end, (255, 0, 0), 2)

        cv2.imshow("MediaPipe Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        print(f"the detail of mp_image is: {mp_image}")
camera.release()
cv2.destroyAllWindows()
