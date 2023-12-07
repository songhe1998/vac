import os
import cv2
from tqdm import tqdm

root = '../../CSL/gloss-zip/color-gloss/color'
save_root = '../../CSL/gloss-zip/color-gloss/resized'

for c in tqdm(os.listdir(root)):
    video_dir = os.path.join(root, c)
    for v in os.listdir(video_dir):
        v_path = os.path.join(video_dir, v)
        key = v.replace('.mp4','')
        save_folder = os.path.join(save_root, c, key)
        os.makedirs(save_folder)

        cap = cv2.VideoCapture(v_path)

        if not cap.isOpened():
            print('Error: Could not open video file')
        else:
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
                    frame_number += 1
                    save_path = os.path.join(save_folder,f'{frame_number}.jpg')
                    cv2.imwrite(save_path, frame)
                else:
                    break

        cap.release()

        