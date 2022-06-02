"""
Test script.

Author: Hanna L.A.
Date: June 2022
"""
import random
import numpy as np
import cv2
import pytest

from . import TEST_PATH
import personal_trainer.pose_module as pm
import personal_trainer.tools as tools
import personal_trainer.curl_bicep_video as mv
import personal_trainer.quickly_pose_img as qp


@pytest.fixture()
def test_image():
    # RGB test_image
    return cv2.imread(str(TEST_PATH / 'data/test_img.jpg'))[:, :, ::-1]


@pytest.fixture()
def mock_image():
    return np.zeros(9).reshape(1, -1, 3).astype(np.uint8)


def test_pose_estimation_simple(test_image):
    detector = pm.PoseEstimation()
    img = detector.find_pose(test_image)
    assert img.any()

    lmk_list = detector.get_landmarks(img)
    assert lmk_list
    assert len(lmk_list) == 33


def test_pose_estimation_invalid(mock_image):
    detector = pm.PoseEstimation()
    with pytest.raises(AssertionError,
                       match='It has been impossible to find landmarks in the test_image'):
        detector.find_pose(mock_image)
    with pytest.raises(AssertionError,
                       match='It has been impossible to find landmarks in the test_image'):
        detector.get_landmarks(mock_image)


def test_pose_estimation_angle():
    img = cv2.imread(str(TEST_PATH / 'data/angle_test.jpg'))[:, :, ::-1]
    detector = pm.PoseEstimation()
    img = detector.find_pose(img, False)
    detector.get_landmarks(img)
    # Left arm
    angle = detector.find_angle(img, 11, 13, 15)
    assert img.any()
    assert round(angle) == 277

    angle = detector.find_angle(img, 11, 13, 15, False)
    assert img.any()
    assert round(angle) == 277


def test_n_resize(test_image):
    h, w, c = test_image.shape
    n = random.randint(1, 5)
    new_size = int(h / n), int(w / n), c
    assert new_size == tools.n_resize(test_image, n).shape


def test_curl_bicep_video_invalid_cap():
    with pytest.raises(Exception, match='Warning: unable to open video source'):
        mv.main('fake_path.mp4', 'right')


def test_curl_bicep_video_invalid_arm():
    with pytest.raises(AssertionError, match='Arm evaluated only can be right or left'):
        mv.main('fake_path.mp4', 'fake_arm')


def test_curl_bicep_video():
    mv.main(str(TEST_PATH / 'data/test_video.mp4'),
            'left')


def test_quickly_pose_img():
    qp.main(str(TEST_PATH / 'data/test_img.jpg'))
