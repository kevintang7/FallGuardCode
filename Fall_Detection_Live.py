import sys
# If necessary, add your site-packages path in Windows so Python can find mediapipe and other packages:
# sys.path.append("C:/Users/Kevin/AppData/Roaming/Python/Python311/site-packages")

import cv2
import mediapipe as mp
import time
import os
from statistics import median
from collections import deque

# For emailing
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# ---------------------------
# Setup MediaPipe and Helpers
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# ---------------------------
# Utility Function
# ---------------------------
def is_standing_or_lying_down(landmarks, frame_width, frame_height):
    """
    Calculate bounding box and ratio to help distinguish lying down vs standing.
    """
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

    x_coords = [landmarks[landmark.value].x for landmark in relevant_landmarks]
    y_coords = [landmarks[landmark.value].y for landmark in relevant_landmarks]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    width = (max_x - min_x) * frame_width
    height = (max_y - min_y) * frame_height

    return (min_x, min_y, max_x, max_y), (height / width)

# ---------------------------
# Email Sending Function
# ---------------------------
def send_email_with_image(
    sender_email,
    sender_password,
    recipient_email,
    subject,
    body_text,
    image_path,
    smtp_server="smtp.gmail.com",
    smtp_port=587
):
    """
    Sends an email with an attached image via SMTP (Gmail).
    """
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject

    # Attach the body text
    message.attach(MIMEText(body_text, "plain"))

    # Attach the image
    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="{os.path.basename(image_path)}"',
    )
    message.attach(part)

    # Convert message to a string
    text = message.as_string()

    # Send the email
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls(context=context)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, text)

# ---------------------------
# Real-time Fall Detection
# ---------------------------
def main():
    # IMPORTANT: For a laptop, change "/dev/video0" (Raspberry Pi camera)
    # to "0" (default integrated or USB webcam).
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Replace these with your actual Gmail address and App Password
    SENDER_EMAIL = "your_sender_address@gmail.com"
    SENDER_PASSWORD = "your_app_password"  # 16-char app password
    RECIPIENT_EMAIL = "recipient_address@gmail.com"

    fall_detected = False
    fall_logged = False
    potential_fall_detected = False
    fall_timestamp = None

    # NEW: Track whether we've captured & emailed the image
    image_sent = False

    fall_threshold = 88
    box_ratio_threshold = 0.555
    presence_threshold = 5

    presence_counter = 0
    previous_ankle = deque(maxlen=presence_threshold)
    previous_shoulder = deque(maxlen=presence_threshold)

    min_box_ratio = float("inf")
    prev_frame_time = 0

    print("Starting real-time fall detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        new_frame_time = time.time()

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            presence_counter += 1

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            h, w, _ = frame.shape
            landmarks = results.pose_landmarks.landmark

            # Coordinates of left shoulder and left ankle in the frame
            shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h,
            ]
            ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h,
            ]

            (min_x_norm, min_y_norm, max_x_norm, max_y_norm), bounding_box_ratio = is_standing_or_lying_down(landmarks, w, h)

            # Convert normalized coords (0-1) to pixel coords
            min_x, max_x = int(min_x_norm), int(max_x_norm)
            min_y, max_y = int(min_y_norm), int(max_y_norm)

            # Draw bounding box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

            if bounding_box_ratio < min_box_ratio:
                min_box_ratio = bounding_box_ratio

            # Keep track of the last N shoulder/ankle positions
            previous_ankle.append(ankle)
            previous_shoulder.append(shoulder)

            # After enough frames, check velocity
            if len(previous_shoulder) >= presence_threshold and presence_counter >= presence_threshold:
                differences_shoulder = [
                    previous_shoulder[i][1] - previous_shoulder[i - 1][1]
                    for i in range(-1, -presence_threshold, -1)
                ]
                differences_shoulder_ankle = [
                    previous_ankle[i][1] - previous_shoulder[i - 1][1]
                    for i in range(-1, -presence_threshold, -1)
                ]

                median_difference = median(differences_shoulder)
                shoulder_ankle_difference = median(differences_shoulder_ankle)

                frame_duration = new_frame_time - prev_frame_time
                fps = 1.0 / frame_duration if frame_duration > 0 else 0

                # Velocity scaled relative to frame height
                vertical_velocity = median_difference * fps * 200 / h

                # Condition to suspect a fall
                if (bounding_box_ratio < box_ratio_threshold or shoulder_ankle_difference < 0) and not fall_logged:
                    potential_fall_detected = True

                # Confirm the fall if velocity threshold is exceeded
                if potential_fall_detected:
                    if vertical_velocity > fall_threshold:
                        fall_detected = True
                        fall_logged = True
                        fall_timestamp = time.time()
                        image_sent = False  # We'll send the image 3 seconds later

            prev_frame_time = new_frame_time

        # If fall is detected, overlay warning text
        if fall_detected:
            cv2.putText(
                frame,
                "Fall Detected!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            # After 3 seconds from detection, if we haven't yet sent the image, do it now
            if not image_sent and fall_timestamp is not None:
                if (time.time() - fall_timestamp) >= 3:
                    # Capture the current frame and email it
                    timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
                    image_filename = f"fall_detected_{timestamp_str}.jpg"
                    cv2.imwrite(image_filename, frame)

                    email_subject = "Fall Detected!"
                    email_body = (
                        f"A fall was detected at {timestamp_str}.\n"
                        "Picture taken 3 seconds after the fall.\n"
                        "See attached image for details."
                    )

                    try:
                        send_email_with_image(
                            sender_email=SENDER_EMAIL,
                            sender_password=SENDER_PASSWORD,
                            recipient_email=RECIPIENT_EMAIL,
                            subject=email_subject,
                            body_text=email_body,
                            image_path=image_filename
                        )
                        print("Fall alert email sent (3 seconds after detection).")
                    except Exception as e:
                        print("Error sending email:", e)

                    image_sent = True

            # Hide the message and reset states after 10 seconds
            if fall_timestamp is not None:
                if (time.time() - fall_timestamp) > 10:
                    fall_detected = False
                    fall_logged = False
                    potential_fall_detected = False
                    fall_timestamp = None
                    image_sent = False

        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting program.")

if __name__ == "__main__":
    main()
