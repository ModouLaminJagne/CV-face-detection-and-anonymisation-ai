# Read Image
import os
import cv2
import numpy as np
import mediapipe as mp
import argparse
import os

# function to process image
def process_img(img, face_detection):

    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    # print(out.detections)
    if out.detections is not None:

        for detection in out.detections:

            location_data = detection.location_data
            bboxC = location_data.relative_bounding_box

        x1, y1, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height
        x1, y1, w, h = int(x1 * W), int(y1 * H), int(w * W), int((h * H)+20)

        img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))
        
    return img
    
args = argparse.ArgumentParser()

# Input mode: image, video, webcam
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)

args = args.parse_args()

# Create output directory
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Detect Image
mp_face_detection = mp.solutions.face_detection

# Initialize Face Detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:

    # Image Mode
    if args.mode in ['image']:

        # Read Image
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detection)
        
        cv2.imwrite(os.path.join(output_dir, "blurred_pic.jpg"), img)
        print("Image saved to:", os.path.join(output_dir, "blurred_pic.jpg"))

    # Video Mode
    elif args.mode in ["video"]:

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_vid = cv2.VideoWriter(os.path.join(output_dir, "blurred_vid.mp4"),
                                     cv2.VideoWriter.fourcc(*'MP4V'),
                                     25,
                                     (frame.shape[1], frame.shape[0]))
        print("Video saved to:", os.path.join(output_dir, "blurred_vid.mp4"))

        # Process Video
        while ret:
            frame = process_img(frame, face_detection)

            output_vid.write(frame)

            ret, frame = cap.read()

        cap.release()
        output_vid.release()

    # Webcam Mode
    elif args.mode in ["webcam"]:

        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()

        # Process Video
        while ret:
            frame = process_img(frame, face_detection)

            cv2.imshow('frame', frame)
            cv2.waitKey(25)

            ret, frame = cam.read()

        cam.release()