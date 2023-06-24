import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculateAngle(a, b, c):
    x1 = a.x
    x2 = b.x
    x3 = c.x
    y1 = a.y
    y2 = b.y
    y3 = c.y
    radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# def vakrasana1(landmarks):
#     t = 0
#     if left_hip_angle >= 85 and right_hip_angle >= 85:
#         t = t + 1
#     else:
#         print("please be straight and sit straight ")
#     if left_elbow_angle >= 165:
#         t = t + 1
#     else:
#         print("Keep your hands on ground and take its support.")
#     if left_shoulder_angle <= 30:
#         t = t + 1
#     else:
#         print("Bring hands near to the body")
#     if (
#         left_knee_angle <= 185
#         and left_knee_angle >= 160
#         and right_knee_angle <= 185
#         and right_knee_angle >= 160
#     ):
#         t = t + 1
#     else:
#         print("Bring the feet near to the body")

#     if t == 4:
#         print("you are doing great")
#     else:
#         print("Please take a look at photo and get in right position")


# def vakrasana2(landmarks):
#     t = 0
#     if left_hip_angle >= 85 and right_hip_angle >= 85:
#         t = t + 1
#     else:
#         print("please be straight and sit straight ")
#     if left_elbow_angle >= 165:
#         t = t + 1
#     else:
#         print("Keep your hands on ground and take its support.")
#     if left_shoulder_angle <= 20:
#         t = t + 1
#     else:
#         print("Bring hands near to the body")
#     if left_knee_angle <= 60:
#         t = t + 1
#     else:
#         print("Bend the knee properly")

#     if t == 4:
#         print("you are doing great")
#     else:
#         print("Please take a look at photo and get in right position")


def vakrasana3(landmarks):
    t = 0
    if left_hip_angle <= 135 and right_hip_angle >= 85:
        t = t + 1
    else:
        print("please be straight and sit straight ")
    if left_elbow_angle >= 165:
        t = t + 1
    else:
        print("Keep your hands on ground and take its support.")
    if left_shoulder_angle <= 60 and left_shoulder_angle >= 20:
        t = t + 1
    else:
        print("Bring hands near to the body")
    if (
        left_knee_angle <= 60
    ):  # and (right_knee_angle >=175 and right_knee_angle <=185)):
        t = t + 1
    else:
        print("Bend the knee properly")

    if t == 4:
        print("you are doing great")
    else:
        print("Please take a look at photo and get in right position")


imageUrl = "python\dummy2.jpg"
# Load the image
image = cv2.imread(imageUrl)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Recolor image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    landmarks = results.pose_landmarks.landmark

    # Get the angle between the left hip, knee, and ankle points.
    left_elbow_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
    )

    # Get the angle between the right shoulder, elbow and wrist points.
    right_elbow_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
    )

    # Get the angle between the left elbow, shoulder and hip points.
    left_shoulder_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
    )

    # Get the angle between the right hip, shoulder and elbow points.
    right_shoulder_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    )

    # Get the angle between the left hip, knee and ankle points.
    left_knee_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
    )

    # Get the angle between the right hip, knee and ankle points
    right_knee_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    )
    # Get the angle between the nose left hip and left ankle
    left_hip_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.NOSE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
    )
    # Get the angle between the nose right hip and right ankle

    right_hip_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.NOSE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    )
    left_hip_s_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
    )
    right_hip_s_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
    )
    # if left_shoulder_angle >= 30:
    vakrasana3(landmarks)
# else:
#     if left_knee_angle <= 60:
#         vakrasana2(landmarks)
#     else:
#         vakrasana1(landmarks)

# Render detections
mp_drawing.draw_landmarks(
    image,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
