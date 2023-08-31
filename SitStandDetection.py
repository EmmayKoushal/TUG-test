import cv2
import numpy
import mediapipe

def caliculate_angle(a, b, c):
    a = numpy.array(a)
    b = numpy.array(b)
    c = numpy.array(c)

    radians = numpy.arctan2(c[1] - b[1], c[0] - b[0]) - numpy.arctan2(a[1] - b[1], a[0] - b[0])
    angle = numpy.abs(radians*180.0/numpy.pi)

    return angle

mp_drawing = mediapipe.solutions.drawing_utils
mp_pose = mediapipe.solutions.pose

#img = cv2.imread('./image3.jpg')
cap = cv2.VideoCapture('./Video.mp4')
print("Check Point 1")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        print("Check Point 2")
        ret, img = cap.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark 
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = caliculate_angle(left_hip, left_knee, left_ankle)
            print("Angle is: ", angle)

            if angle > 120:
                cv2.putText(image, 'Pose: Standing', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            else:
                cv2.putText(image, 'Pose: Sitting', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3))

        

        cv2.imshow('MediaPipe Streem', image)

        if cv2.waitKey(1) == 13:
            break

cap.release()
cv2.destroyAllWindows()