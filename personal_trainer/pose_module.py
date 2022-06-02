"""
PostEstimation class

Author: Hanna L.A.
Date: June 2022
"""
from typing import List
import math
import numpy as np
import cv2
import mediapipe as mp


class PoseEstimation:
    """
    Main class for pose estimation.
    """

    def __init__(self,
                 static_image_mode=False,
                 smooth=True):
        """

        Args:
            static_image_mode:  Whether to treat the input images as a batch of static
                                and possibly unrelated images, or a video stream. See details in
            smooth: Whether to filter landmarks across different input
                    images to reduce jitter
        """

        # Basic
        self.static_image_mode = static_image_mode
        self.smooth = smooth

        # mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # Set pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.static_image_mode,
                                      smooth_landmarks=self.smooth)

        # Output
        self.results = None
        self.list_landmarks = None

    def find_pose(self,
                  img: np.ndarray,
                  draw: bool = True) -> np.ndarray:
        """
        Process pose in an image.

        Args:
            img: A three channel RGB image represented as numpy ndarray.
            draw: Whether to draw landmarks.

        Returns:
            Image.

        """
        # Pose landmarks
        img = np.ascontiguousarray(img, dtype=np.uint8)
        self.results = self.pose.process(img)
        assert self.results.pose_landmarks,\
            'It has been impossible to find landmarks in the test_image'

        # Draw
        if draw:
            self.mp_drawing.draw_landmarks(img,
                                           self.results.pose_landmarks,
                                           self.mp_pose.POSE_CONNECTIONS)
            return img
        return img

    def get_landmarks(self,
                      img: np.ndarray) -> List:
        """
        List of landmarks in an image.

        Args:
            img: A three channel RGB test_image represented as numpy ndarray.

        Returns:
            Landmarks list.

        """
        assert self.results.pose_landmarks, \
            'It has been impossible to find landmarks in the test_image'

        # Landmarks name (33 points)
        names = [self.mp_pose.PoseLandmark(i).name for i in range(33)]

        # Landmarks list
        self.list_landmarks = []
        for name, (idx, landmark) in zip(names, enumerate(self.results.pose_landmarks.landmark)):
            height, width, _ = img.shape
            x_centroid = int(landmark.x * width)
            y_centroid = int(landmark.y * height)
            self.list_landmarks.append([idx,
                                        name,
                                        x_centroid,
                                        y_centroid])
        return self.list_landmarks

    def find_angle(self,
                   img: np.ndarray,
                   p1: int,
                   p2: int,
                   p3: int,
                   draw: bool = True) -> float:
        """
        Find an angle between two lines build with three points.

        Args:
            img: A three channel RGB test_image represented as numpy ndarray.
            p1: Point 1.
            p2: Point 2. Intersection.
            p3: Point 3.
            draw: Whether to draw landmarks.

        Returns:
            Angle.

        """
        img = np.ascontiguousarray(img, dtype=np.uint8)

        # Get the landmarks
        x1, y1 = self.list_landmarks[p1][2:]
        x2, y2 = self.list_landmarks[p2][2:]
        x3, y3 = self.list_landmarks[p3][2:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2,
                                        x3 - x2)
                             - math.atan2(y1 - y2,
                                          x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        xs = [x1, x2, x3]
        ys = [y1, y2, y3]
        if draw:
            for x, y in zip(xs, ys):
                cv2.circle(img, (x, y), 10, 255, cv2.FILLED)
            color = (255, 255, 255)
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
            cv2.line(img, (x3, y3), (x2, y2), color, 3)
            cv2.rectangle(img, (x2 - 50, y2 + 20), (x2 + 10, y2 + 60), color, cv2.FILLED)
            cv2.putText(img, f'{angle:.0f}', (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            return angle
        return angle
