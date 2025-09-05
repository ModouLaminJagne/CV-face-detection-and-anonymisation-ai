# Read Image
import os
import cv2
import numpy as np
import mediapipe as mp
import os


output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Read Image
img_path = "C:\\Users\\Administrator\\Documents\\My Work\\CV-face-detection-and-anonymisation-ai\\IMG_4296.jpg"
img = cv2.imread(img_path)

H, W, _ = img.shape

# Detect Image
mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    # print(out.detections)
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bboxC = location_data.relative_bounding_box

        x1, y1, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height
        x1, y1, w, h = int(x1 * W), int(y1 * H), int(w * W), int((h * H)+20)

        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 20)

        # Blur Image
        Bimg = img.copy()
        Bimg[y1:y1 + h, x1:x1 + w, :] = cv2.blur(Bimg[y1:y1 + h, x1:x1 + w, :], (70, 70))
        # Bimg[y1:y1 + (h-5), x1:x1 + w, :] = cv2.GaussianBlur(Bimg[y1:y1 + (h-5), x1:x1 + w, :], (50, 50), 30)
    else:
        Bimg = img.copy()
        print("No face detected")

    # cv2.imshow("Image", Bimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# Save Image
# cv2.imwrite("C:\\Users\\Administrator\\Desktop\\111_blurred2.jpg", Bimg)
cv2.imwrite(os.path.join(output_dir, "blurred_image2.jpg"), Bimg)
print("Image saved to:", os.path.join(output_dir, "blurred_image2.jpg"))