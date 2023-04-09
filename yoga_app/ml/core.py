import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from matplotlib.pyplot import imshow
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import landmark_pb2
import enum


class Stage(str,enum.Enum):
    t_pose = "t_pose"
    one = "one"
    two = "two"
    three = "three"
    four = "four"
    five = "five"
    six = "six"
    seven = "seven"

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

    # if cv2.waitKey(10) & 0xFF == ord('q'):
        #    break

        #cap.release()
        #cv2.destroyAllWindows()

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
        #mp_drawing.draw_landmarks(image, landmark_subset,
                            #       mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            #       mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                            #       )               
        mp_drawing.draw_landmarks(
        image,
        landmark_list=landmark_subset)
        #imshow(annotated_image)    
        imshow(image)

    # if cv2.waitKey(10) & 0xFF == ord('q'):
        #    break

        #cap.release()
        #cv2.destroyAllWindows()

    len(landmarks)

    #results.pose_landmark

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

    # Get the angle between the left shoulder, elbow and wrist points. 
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


    pose_stage = None
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the both arms are straight.
    if (left_elbow_angle>=165 and left_elbow_angle<=195) and (right_elbow_angle >=165 and right_elbow_angle<= 195):
 
        # Check if shoulders are at the required angle.
        if (left_shoulder_angle>=75 and left_shoulder_angle<= 105) and (right_shoulder_angle>= 75 and right_shoulder_angle<= 105):
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
            # Check if both legs are straight
            if (left_knee_angle >=165 and left_knee_angle <= 195) and (right_knee_angle >= 165 and right_knee_angle <= 195):
 
                # Specify the label of the pose that is tree pose.
                #label = 'T Pose'
                pose_stage = Stage.t_pose


      # To check if it is surya namaskar pose 1

    if (left_hip_angle >=160 and right_hip_angle >=160) and (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x< landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x):
      
      if(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y >landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y ) and (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y> landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y):
        if(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x-landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x<=0.1):

          pose_stage = Stage.one



   #check if it is surya namaskar pose 2(move 90 degree toward your right)


    if((left_elbow_angle>120 and right_elbow_angle>120) and (left_knee_angle>150 and right_knee_angle>150)):
     
     if((landmarks[mp_pose.PoseLandmark.NOSE.value].y>landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y) and (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x >landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x)):

       if(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x<=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x):

         pose_stage = Stage.two


    

    #to check if it is surya namaskar pose 3(move 90 degree to your right)


    if((left_knee_angle>160 and right_knee_angle>160) and (left_hip_angle<90 and right_hip_angle<90)):
       if((landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y>landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y) and (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y<landmarks[mp_pose.PoseLandmark.NOSE.value].y)):
            pose_stage = Stage.three

    #to check if it is surya namsakr pose 4(move 90 degrees to left and left leg forward)
   
    if((left_knee_angle <90 and right_hip_s_angle>150) and (right_knee_angle>140 and left_elbow_angle>150)):
      if((landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y<landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y) and (landmarks[mp_pose.PoseLandmark.NOSE.value].y<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)):
        if(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y<landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y):
          pose_stage = Stage.four

    #for pose 5
    if((left_hip_angle>150 and right_hip_angle>150) and (left_knee_angle>150 and right_knee_angle>150)):
      if(right_shoulder_angle<90 and left_shoulder_angle<90):
        if((landmarks[mp_pose.PoseLandmark.NOSE.value].y<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y) and (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y<landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)):
          pose_stage = Stage.five
    #for pose 6
    if((left_hip_s_angle<90 and right_hip_s_angle<90) and (left_knee_angle>90 and right_knee_angle>90)):
      if(landmarks[mp_pose.PoseLandmark.NOSE.value].y>landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y):
        pose_stage = Stage.six
    #for pose 7
    if((left_knee_angle>150 and right_knee_angle>150) and (left_hip_s_angle>120 and right_hip_s_angle>120)):
      if((landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y>landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y) and (landmarks[mp_pose.PoseLandmark.NOSE.value].y<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)):
        pose_stage = Stage.seven

    #for pose 8
    
    print("pose_stage------->",pose_stage)
    suggestions = []
    if pose_stage == Stage.one:
        t = 0
        if (left_hip_angle >=160 and right_hip_angle >=160):
            t =t+1
        else:
            print("pls be straight and stand straight ")
            suggestions.append("pls be straight and stand straight ")
        if (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x< landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x):
            t = t+1
        else:
            print("Pls put your hands in centre")
            suggestions.append("Pls put your hands in centre")
        if ((landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y >landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y )):
            t = t+1
        else:
                print("pls put your hands above your waist to near your chest")
                suggestions.append("pls put your hands above your waist to near your chest")
        if(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y> landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y):
            t = t+1
        else:
            print("pls put your hands below your shoulder ")
            suggestions.append("pls put your hands below your shoulder ")
        if (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x-landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x<=0.1): 
            t = t+1
        else:
            print("pls put your hand together to make a NAMASKAR pose from hands")
            suggestions.append("pls put your hand together to make a NAMASKAR pose from hands")
        if t==5:
            print("you are doing great")
            suggestions.append("you are doing great")
        else:
            print("Please take a look at photo and get in right position")
            suggestions.append("Please take a look at photo and get in right position")
            

    if pose_stage == Stage.two:
        t = 0
        if(left_elbow_angle>120 and right_elbow_angle>120):
            t = t+1
        else:
            print("please straighten your arms behind your head")
            suggestions.append("please straighten your arms behind your head")
        if(left_knee_angle>150 and right_knee_angle>150):
            t =t+1
        else:
            print("please don't bend your knees keep them straight")
            suggestions.append("please don't bend your knees keep them straight")
        if((landmarks[mp_pose.PoseLandmark.NOSE.value].y>landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)):
            t =t+1
        else:
            print("please keep your wrist above your head in straight condition")
            suggestions.append("please keep your wrist above your head in straight condition")
        if (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x >landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x):
            t =t+1
        else:
            print("please push back your hands harder")
            suggestions.append("please push back your hands harder")
            
        if (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x<=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x):
            t = t+1
        else:
            print("please bend a bit more backwards ")
            suggestions.append("please bend a bit more backwards ")

        if t==5:
            print("You are in right position ")
            suggestions.append("You are in right position ")
        else:
            print("Please take a look at photo and get in right position")
            suggestions.append("Please take a look at photo and get in right position")


    if pose_stage == Stage.three:
        t =0
        if((right_knee_angle >140 and right_hip_s_angle>150)):
            t =t+1
        else:
            print("Please keep your right leg straight and make your ankle touch ground")
            suggestions.append("Please keep your right leg straight and make your ankle touch ground")
        if left_knee_angle <90 :
            t =t+1
        else:
            print("Bend your left knee more ")
            suggestions.append("Bend your left knee more ")
        if left_elbow_angle>150 and right_elbow_angle >150:
            t = t+1
        else:
            print("please keep your hands bit more straight")
            suggestions.append("please keep your hands bit more straight")
        if (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y<landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y):
            t =t+1
        else:
            print("please keep your waist down and in line with legs")
            suggestions.append("please keep your waist down and in line with legs")
        if ((landmarks[mp_pose.PoseLandmark.NOSE.value].y<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)):
            t =t+1
        else:
            print("look up or keep your head up")
            suggestions.append("look up or keep your head up")
        if (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y<landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y):
            t =t+1
        else:
            print("Your right knee should touch the ground ")
            suggestions.append("Your right knee should touch the ground ")

        if t==6:
            print("you are doing great")
            suggestions.append("you are doing great")
            
    return suggestions
    



