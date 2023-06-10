import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from google.colab.patches import cv2_imshow
from mediapipe.framework.formats import landmark_pb2

class Stage(str,enum.Enum):
    one = "one"
    two = "two"
    three = "three"

def refactor_this_later(image_path):

    suggestions = []
    cap = cv2.VideoCapture(image_path)
## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    #while cap.isOpened():
      ret, frame = cap.read()
        
        # Recolor image to RGB
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      
        # Make detection
      results = pose.process(image)
    
        # Recolor back to BGR
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # Extract landmarks
      landmarks = results.pose_landmarks.landmark
      #print(landmarks)
        
        # Render detections
      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
               
      print(type(results.pose_landmarks))    
      imshow(image)

    tup1 = ((0,11),(11,12),(12,14),(14,16),(11,13),(13,15),(11,23),(12,24),(23,24),(24,26),(23,25),(25,27),(26,28))
    fnum = frozenset(tup1)
    fnum

    cap = cv2.VideoCapture(image_path)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    #while cap.isOpened():
    ret, frame = cap.read()
        
        # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
      
        # Make detection
    results = pose.process(image)
    
        # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # Extract landmarks
    landmarks = results.pose_landmarks.landmark
      #print(landmarks)
    landmark_subset = landmark_pb2.NormalizedLandmarkList(
      landmark = [
          results.pose_landmarks.landmark[0],
          results.pose_landmarks.landmark[11], 
          results.pose_landmarks.landmark[12],
          results.pose_landmarks.landmark[13],
          results.pose_landmarks.landmark[14],
          results.pose_landmarks.landmark[15],
          results.pose_landmarks.landmark[16],
          results.pose_landmarks.landmark[23],
          results.pose_landmarks.landmark[24],
          results.pose_landmarks.landmark[25],
          results.pose_landmarks.landmark[26],
          results.pose_landmarks.landmark[27],
          results.pose_landmarks.landmark[28],
      ]
    )
    #print(landmark_subset)
        # Render detections
    # mp_drawing.draw_landmarks(image, landmark_subset,
    #                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
    #                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
    #                             )               
    mp_drawing.draw_landmarks(
    image,
    landmark_list=landmark_subset)
    #cv2_imshow(annotated_image)    
    imshow(image)

    #results.pose_landmark
   len(landmarks)
   landmark_subset

   landmark_subset = landmark_pb2.NormalizedLandmarkList(
      landmark = [
          results.pose_landmarks.landmark[13],
          results.pose_landmarks.landmark[14], 
          results.pose_landmarks.landmark[25],
          results.pose_landmarks.landmark[26], 
      ]
)
      annotated_image = image.copy()
      mp_drawing.draw_landmarks(
      image=annotated_image,
      landmark_list=landmark_subset)
      imshow(annotated_image)

      def calculate_Angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
      return angle

    def calculateAngle(a,b,c):
      x1=a.x
      x2=b.x
      x3=c.x
      y1=a.y
      y2=b.y
      y3=c.y
      radians = np.arctan2(y3-y2, x3-x2) - np.arctan2(y1-y2, x1-x2)
      angle = np.abs(radians*180.0/np.pi)
        
      if angle >180.0:
          angle = 360-angle
            
            
      return angle

left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
# Get the angle between the right shoulder, elbow and wrist points. 
right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
 
    # Get the angle between the right hip, shoulder and elbow points. 
right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
 
    # Get the angle between the left hip, knee and ankle points. 
left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
 
    # Get the angle between the right hip, knee and ankle points 
right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
     #Get the angle between the nose left hip and left ankle
left_hip_angle=calculateAngle(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
     #Get the angle between the nose right hip and right ankle
    
right_hip_angle=calculateAngle(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
left_hip_s_angle=calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
right_hip_s_angle=calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])

hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
angle = calculate_Angle(hip, knee, ankle)
print(angle)

shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
angle = calculate_Angle(shoulder, elbow, wrist)
print(angle)

for i in range(2):
        print(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value])

        pose_stage=None

        if(left_shoulder_angle>=30):
          pose_stage=Stage.three
          
          #vakrasana3(landmarks)
        else:
          if(left_knee_angle<=60):
            pose_stage=Stage.two

            #vakrasana2(landmarks)
          else:
            pose_stage=Stage.one

            #vakrasana1(landmarks)

      print("pose_stage------->",pose_stage)
      suggestions = []
  
            #def vakrasana1(landmarks):

if pose_stage==Stage.one:   
    t = 0
    if (left_hip_angle >=85 and right_hip_angle >=85):
      t =t+1
    else:
      print("please be straight and sit straight ")
      suggestions.append("please be straight and sit straight ")
    if (left_elbow_angle >=165):
      t=t+1
    else:
      print("Keep your hands on ground and take its support.")
      suggestions.append("Keep your hands on ground and take its support.")
    if(left_shoulder_angle <=30):
      t=t+1
    else:
      print("Bring hands near to the body")
      suggestions.append("Bring hands near to the body")
    if( (left_knee_angle <=185 and left_knee_angle>=160 and right_knee_angle <=185 and right_knee_angle>=160)):
      t=t+1
    else:
      print("Bring the feet near to the body")
      suggestions.append("Bring the feet near to the body")
          
    if t==4:
      print("you are doing great")
      suggestions.append("you are doing great")
    else:
      print("Please take a look at photo and get in right position")
      suggestions.append("Please take a look at photo and get in right position")
        
#vakrasana1(landmarks)

            #def vakrasana2(landmarks):
if pose_stage==Stage.two:   
    t = 0
    if (left_hip_angle >=85 and right_hip_angle >=85):
      t =t+1
    else:
      print("please be straight and sit straight ")
      suggestions.append("please be straight and sit straight ")
    if (left_elbow_angle >=165):
      t=t+1
    else:
      print("Keep your hands on ground and take its support.")
      suggestions.append("Keep your hands on ground and take its support.")
    if(left_shoulder_angle <=20):
      t=t+1
    else:
      print("Bring hands near to the body")
      suggestions.append("Bring hands near to the body")
    if(left_knee_angle <=60):
      t=t+1
    else:
      print("Bend the knee properly")
      suggestions.append("Bend the knee properly")
          
    if t==4:
      print("you are doing great")
      suggestions.append("you are doing great")
    else:
      print("Please take a look at photo and get in right position")
      suggestions.append("Please take a look at photo and get in right position")
        
#vakrasana2(landmarks)

            #def vakrasana3(landmarks):
if pose_stage==Stage.three:   
    t = 0
    if (left_hip_angle <=135 and right_hip_angle >=85):
      t =t+1
    else:
      print("please be straight and sit straight ")
      suggestions.append("please be straight and sit straight ")
    if (left_elbow_angle >=165):
      t=t+1
    else:
      print("Keep your hands on ground and take its support.")
      suggestions.append("Keep your hands on ground and take its support.")
    if(left_shoulder_angle <=60 and left_shoulder_angle>=20):
      t=t+1
    else:
      print("Bring hands near to the body")
      suggestions.append("Bring hands near to the body")
    if(left_knee_angle <=60):# and (right_knee_angle >=175 and right_knee_angle <=185)):
      t=t+1
    else:
      print("Bend the knee properly")
      suggestions.append("Bend the knee properly")
          
    if t==4:
      print("you are doing great")
      suggestions.append("you are doing great")
    else:
      print("Please take a look at photo and get in right position")
      suggestions.append("Please take a look at photo and get in right position")
        
return suggestions




