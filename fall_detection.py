import sys
#sys.path.append("C:/Users/Kevin/AppData/Roaming/Python/Python311/site-packages")
from statistics import median
import cv2
import mediapipe as mp
import os
import time
from collections import deque

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
#This is setup, the 0.7 means that it has to be a 70% confidence inorder to count a pose point.
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize MediaPipe drawing
mp_drawing = mp.solutions.drawing_utils
    
# Function to check if a person is standing or lying down and get bounding box coordinates
def is_standing_or_lying_down(landmarks, frame_width, frame_height):
    relevant_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    ]
    
    x_coords = [landmarks[landmark].x for landmark in relevant_landmarks]
    y_coords = [landmarks[landmark].y for landmark in relevant_landmarks]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width = (max_x - min_x) * frame_width
    height = (max_y - min_y) * frame_height
    
    
    return (min_x, min_y, max_x, max_y), height / width

#used to store data, could be used.
velocity_data = {}
bounding_data = {} 
# Process each video in the directory
def process_video(video_path):
    global output_csv

    # Define output video path
    video_name = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # Define the codec and create VideoWriter object, not used right now.
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    velocity_list = [] 
    bounding_list = []

    fall_detected = False
    fall_logged = False

    frame_count = 0
    fall_threshold = 88 # Updated threshold for vertical velocity
    box_ratio_threshold = 0.555
    max_vertical_velocity = 0
    overall_max_vertical_velocity = 0
    presence_counter = 0
    presence_threshold = 5
    min_box_ratio = float('inf')
    potential_fall_detected = False
    shoulder_ankle_difference = 1
    vertical_velocity = 0
    bounding_box_ratio = 100
    previous_ankle = deque(maxlen=presence_threshold) 
    previous_shoulder = deque(maxlen=presence_threshold) 
    prev_frame_time = 0

    while cap.isOpened():
        new_frame_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        start_col = width * 0 // 22
        right_frame = frame[:, start_col:]

        frame_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)        
        
        if results.pose_landmarks:
            presence_counter += 1
            
            mp_drawing.draw_landmarks(
                right_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
            (min_x, min_y, max_x, max_y), bounding_box_ratio = is_standing_or_lying_down(landmarks, width, height)

            min_x = int(min_x * width)
            max_x = int(max_x * width)
            min_y = int(min_y * height)
            max_y = int(max_y * height)

            cv2.rectangle(right_frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

            if bounding_box_ratio < min_box_ratio:
                min_box_ratio = bounding_box_ratio

            if len(previous_shoulder) >= presence_threshold and presence_counter >= presence_threshold:
                
                # Calculate the differences in Y positions

                differencesShoulder = [
                    previous_shoulder[i][1] - previous_shoulder[i - 1][1]
                    for i in range(-1, -1*presence_threshold, -1)  # Last 4 differences
                ]

                differencesShoulderAnkle = [
                    previous_ankle[i][1] - previous_shoulder[i - 1][1]
                    for i in range(-1, -1*presence_threshold, -1)  # Last 4 differences
                ]

                # Compute the median of the differences
                median_difference = median(differencesShoulder)
                shoulder_ankle_difference = median(differencesShoulderAnkle)

                # Calculate vertical velocity
                vertical_velocity = median_difference * frame_rate * 200 / height     

                # Store vertical velocity for the frame
                velocity_list.append(vertical_velocity)
                bounding_list.append(bounding_box_ratio*10)

                # vertical_velocity fabs(height-max_y)                              
                if vertical_velocity > max_vertical_velocity:
                    max_vertical_velocity = vertical_velocity

                if vertical_velocity > overall_max_vertical_velocity:
                    overall_max_vertical_velocity = vertical_velocity                      
               

                if  (bounding_box_ratio < box_ratio_threshold or shoulder_ankle_difference<0) and not fall_logged:
                    potential_fall_detected = True

                if potential_fall_detected: 
                    # Determine how many frames correspond to 1 second
                    frames_to_check = int(frame_rate * 1)  # Convert 1 second into frames

                    if len(velocity_list) >= frames_to_check:
                        max_velocity_before_fall = max(velocity_list[-frames_to_check:])  # Extract max from last 10 frames
                    else:
                        max_velocity_before_fall = max(velocity_list) if velocity_list else 0  # Use available frames
                    
                    if max_velocity_before_fall>fall_threshold:
                        fall_detected = True
                        fall_logged = True

            previous_ankle.append(ankle)
            previous_shoulder.append(shoulder)
            frame_count += 1
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

        #out.write(frame)
        frame_count += 1

        if fall_detected:
            cv2.putText(frame, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    velocity_data[video_name] = velocity_list
    bounding_data[video_name] = bounding_list

# Recursively read videos from folders
def read_videos_from_folders(base_path):
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                process_video(video_path)
                
if __name__ == "__main__":

    '''base_folders = [
        r"C:\\Users\\homePC\\Downloads\\FallGuard\\archive\\Lecture_room\\Lecture_room\\Videos",
        r"C:\\Users\\homePC\\Downloads\\FallGuard\\archive\\Home_01\\Home_01\\Videos",
        r"C:\\Users\\homePC\\Downloads\\FallGuard\\archive\\Home_02\\Home_02\\Videos",
        r"C:\\Users\\homePC\\Downloads\\FallGuard\\archive\\Coffee_room_01\\Coffee_room_01\\Videos",
        r"C:\\Users\\homePC\\Downloads\\FallGuard\\archive\\Coffee_room_02\\Coffee_room_02\\Videos",
        r"C:\\Users\\homePC\\Downloads\\FallGuard\\archive\\Office\\Office\\Videos",
    ]'''

    base_folders = [
        r"C:\\Users\\homePC\\Downloads\\FallGuard\\Falls",
        r"C:\\Users\\homePC\\Downloads\\FallGuard\\ADL",          
    ]
    for folder in base_folders:
        print(f"Reading videos from: {folder}")
        read_videos_from_folders(folder)