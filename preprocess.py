import dlib
import numpy as np
import cv2

def get_cropped(img):

    # 获取人脸边框
    detector = dlib.get_frontal_face_detector()
    detected_faces = detector(img)

    # 是否检测到人脸
    if len(detected_faces) == 0:
        print('warning: no detected face')
        return None
    loc = detected_faces[0]

    # 人脸是否完整
    if loc.left() < 0 or loc.right() < 0 or loc.top() < 0 or loc.bottom() < 0:
        print('warning: part of face in image')
        return None

    # 裁切图像
    cropped_img = img[loc.top():loc.bottom(), loc.left():loc.right(), :]

    return cropped_img, loc

def get_mask(img, loc):

    # 获取人脸特征点
    predictor_path = 'shape_predictor_81_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    shape = predictor(img, loc)
    landmarks = np.asarray([[p.x, p.y] for p in shape.parts()], dtype=np.int)

    # 提取人脸轮廓
    if landmarks.shape[0] != 81:
        raise Exception(
            'get_image_hull_mask works only with 81 landmarks')

    points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
        12, 13, 14, 15, 16, 78, 74, 79, 73, 72, 80, 
        71, 70, 69, 68, 76, 75, 77]

    outline = []
    for point in points:
        outline.append(landmarks[point])

    outline = np.asarray(outline, dtype=np.int32)

    # 生成binary mask
    mask = np.zeros(img.shape[0:2]+(1,), dtype=np.int32)
    cv2.fillPoly(mask, [outline], (255, 255, 255))

    # 裁减binary mask
    cropped_mask = mask[loc.top():loc.bottom(), loc.left():loc.right(), :]

    return cropped_mask

def preprocess(depth_img, IR_img, aligned_img):
    try:
        cropped_aligned, loc = get_cropped(aligned_img)
        # cv2.imwrite('cropped.jpg', cropped_aligned)
    except TypeError:
        return None

    cropped_depth = depth_img[loc.top():loc.bottom(), loc.left():loc.right(), :]
    cropped_IR = IR_img[loc.top():loc.bottom(), loc.left():loc.right(), :]
    mask = get_mask(aligned_img, loc)

    masked_depth = np.where(mask == 0, 0, cropped_depth)
    masked_IR = np.where(mask == 0, 0, cropped_IR)
    masked_aligned = np.where(mask == 0, 0, cropped_aligned)
    # cv2.imwrite('masked.jpg', masked_aligned)

    return masked_depth, masked_IR
