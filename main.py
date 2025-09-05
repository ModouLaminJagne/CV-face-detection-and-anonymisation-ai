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
img_path = "C:\\Users\\Administrator\\Desktop\\111.jpg"
img = cv2.imread(img_path)

H, W, _ = img.shape

