import os
import numpy as np
import face_alignment
from PIL import Image

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

def align_and_crop(image_path, out_path, output_size=1024, margin=0.25):
    img = cv2.imread(image_path)
    if img is None:
        return False
    pts = fa.get_landmarks(img)
    if pts is None or len(pts)==0:
        return False
    lm = pts[0]
    left = lm[36:42].mean(axis=0)
    right = lm[42:48].mean(axis=0)
    eye_center = (left + right) / 2.0
    nose = lm[30]
    center = ((eye_center + nose) / 2.0).astype(int)
    h, w = img.shape[:2]
    min_x, min_y = lm.min(axis=0).astype(int)
    max_x, max_y = lm.max(axis=0).astype(int)
    box_w = max_x - min_x
    box_h = max_y - min_y
    box_size = int(max(box_w, box_h) * (1 + margin))
    x1 = int(center[0] - box_size//2)
    y1 = int(center[1] - box_size//2)
    x2 = x1 + box_size
    y2 = y1 + box_size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)
    return True
