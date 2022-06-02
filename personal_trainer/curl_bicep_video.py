"""
Module of dumbbell biceps curl to count the correct repetitions.
Author: Hanna L.A.
Date: June 2022
"""
from typing import Union
from pathlib import Path
import numpy as np
import cv2

import personal_trainer.pose_module as pm
from definitions import DATA_PATH


def main(video_path: Union[str, Path],
         arm_evaluated: str) -> None:
    """
    Main function to count bicep curl exercise.

    Args:
        video_path: Video path in data folder.
        arm_evaluated: Choose between left or right to analyze curl bicep exercise.


    """
    assert arm_evaluated.lower() in ['right', 'left'], \
        'Arm evaluated only can be right or left'

    detector = pm.PoseEstimation()
    count = 0
    direction = 0

    cap = cv2.VideoCapture(str(video_path))
    if cap is None or not cap.isOpened():
        raise Exception('Warning: unable to open video source')

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Convert to RGB
        img = img[:, :, ::-1]

        # Resize img by n
        # img = n_resize(img, 3)
        img = cv2.resize(img, (1280, 720))

        # Detector
        img = detector.find_pose(img, False)
        detector.get_landmarks(img)

        # Arm to analyze
        if arm_evaluated == 'right':
            # Right arm
            angle = detector.find_angle(img, 12, 14, 16)
        else:
            # Left arm
            angle = detector.find_angle(img, 11, 13, 15)

        # Percentage: consider that an angle equals 210 is not a curl bicep,
        # and an angle equals to 310 is a perfect curl bicep
        percentage = np.interp(angle, (210, 310), (0, 100))

        # Check bicep curl
        color = (255, 255, 255)
        if percentage == 100:
            color = (127, 61, 127)
            if direction == 0:  # up
                # Count half curl
                count += 0.5
                # Change direction to down
                direction = 1

        if percentage == 0:
            color = (127, 61, 127)
            if direction == 1:  # down
                # Count half curl
                count += 0.5
                # Change direction to up
                direction = 0

        # Draw bar
        height, width, _ = img.shape
        height_bar = int(height / 7)
        width_bar = int(width * 85 / 100)
        bar = np.interp(angle, (220, 310), (height_bar + 500, height_bar))

        cv2.rectangle(img,
                      (width_bar, height_bar),
                      (width_bar + 70, height_bar + 500),
                      color,
                      3)
        cv2.rectangle(img,
                      (width_bar, int(bar)),
                      (width_bar + 70, height_bar + 500),
                      color,
                      cv2.FILLED)

        cv2.putText(img,
                    f'{percentage:.0f} %',
                    (width_bar, int(height * 10 / 100)),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    color,
                    3)

        # Draw curl count
        cv2.rectangle(img,
                      (0, int(height * 80 / 100)),
                      (int(width * 15 / 100), height),
                      (127, 61, 127),
                      cv2.FILLED)
        cv2.putText(img,
                    f'{int(count)}',
                    (int(width * 3 / 100), int(height * 97 / 100)),
                    cv2.FONT_HERSHEY_PLAIN,
                    10,
                    (255, 0, 0),
                    10)

        cv2.imshow('Video', img[:, :, ::-1])  # convert to RGB to use correctly in opencv
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':   # pragma: no cover
    ARM = 'left'
    VIDEO_PATH = DATA_PATH / 'video.mp4'
    main(VIDEO_PATH, ARM)
