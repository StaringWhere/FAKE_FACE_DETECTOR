import dlib
from skimage.transform import estimate_transform, warp, resize
from predictor import PosPrediction
import numpy as np
from scipy import ndimage
import os
import sys
import cv2
sys.path.insert(0,'.')

'''
用于生成深度图的两个函数
'''
def isPointInTri(point, tri_points):
    tp = tri_points

    # vectors
    v0 = tp[:,2] - tp[:,0]
    v1 = tp[:,1] - tp[:,0]
    v2 = point - tp[:,0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)

def render_texture(vertices, colors, triangles, h, w, c = 3):
    ''' render mesh by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width    
    '''
    # initial 
    image = np.zeros((h, w, c))

    depth_buffer = np.zeros([h, w]) - 999999.
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3. 
    tri_tex = (colors[:, triangles[0,:]] + colors[:,triangles[1,:]] + colors[:, triangles[2,:]])/3.

    for i in range(triangles.shape[1]):
        tri = triangles[:, i] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[0,tri]))), 0)
        umax = min(int(np.floor(np.max(vertices[0,tri]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[1,tri]))), 0)
        vmax = min(int(np.floor(np.max(vertices[1,tri]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if tri_depth[i] > depth_buffer[v, u] and isPointInTri([u,v], vertices[:2, tri]): 
                    depth_buffer[v, u] = tri_depth[i]
                    image[v, u, :] = tri_tex[:, i]
    return image

def crop(img):
    detector = dlib.get_frontal_face_detector()
    detected_faces = detector(img)
    if len(detected_faces) == 0:
        print('warning: no detected face')
        return None

    #获取人脸边框坐标
    d = detected_faces[0]
    #d = detected_faces[0].rect
    left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        print('warning: part of face in image')
        return None


    # 计算中心坐标和新的size
    # old_size = (right - left + bottom - top)/2
    # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    # size = int(old_size*1.3)

    # new_left = int(center[0] - size/2)
    # new_right = int(center[0] + size/2)
    # new_top = int(center[1] + size/2)
    # new_bottom = int(center[1] - size/2)

    # loc = [new_bottom, new_top, new_left, new_right]
    loc = [top, bottom, left, right]
    #裁切并保存
    # cropped_img = img[new_bottom:new_top, new_left:new_right, :]
    cropped_img = img[top:bottom, left:right, :]
    return cropped_img, loc

def get_depth(img):
    height = img.shape[0]
    width = img.shape[1]
    img = resize(img, (256,256))
   
    #送入PRnet
    pos_predictor = PosPrediction(256, 256)
    #pos_predictor.restore('/home/mankasto/桌面/FAKE_FACE_DETECTOR/Preprocess/256_256_resfcn256_weight')
    pos_predictor.restore('./256_256_resfcn256_weight')
    pos = pos_predictor.predict(img) #网络推断
    
    #triangles = np.loadtxt('/home/mankasto/桌面/FAKE_FACE_DETECTOR/Preprocess/triangles.txt').astype(np.int32) # ntri x 3
    #face_ind = np.loadtxt( '/home/mankasto/桌面/FAKE_FACE_DETECTOR/Preprocess/face_ind.txt').astype(np.int32) # get valid vertices in the pos map
    triangles = np.loadtxt('./triangles.txt').astype(np.int32) # ntri x 3
    face_ind = np.loadtxt( './face_ind.txt').astype(np.int32) # get valid vertices in the pos map
    all_vertices = np.reshape(pos, [256**2, -1])
    vertices = all_vertices[face_ind, :]
    
    #生成深度图
    z = vertices[:, 2:]
    z = z/max(z)
    depth_img = render_texture(vertices.T, z.T, triangles.T, 256, 256, 1)
    depth_img = np.squeeze(depth_img)
    depth_img = resize(depth_img, (height,width))
    return depth_img

def get_mask(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.array([[0 for i in range(width)]for j in range(height)])
    for i in range(height):
        for j in range(width):
            if img[i][j]!=0:
                mask[i][j]=255
    return mask

# def get_mask(img):

#     # 获取人脸边框
#     detector = dlib.get_frontal_face_detector()
#     detected_faces = detector(img)
#     # 是否检测到人脸
#     if len(detected_faces) == 0:
#         print('warning: no detected face')
#         return None
#     d = detected_faces[0]
#     left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
#     # 人脸是否完整
#     if left < 0 or right < 0 or top < 0 or bottom < 0:
#         print('warning: part of face in image')
#         return None
#     # 边框坐标
#     loc = [top, bottom, left, right]

#     # 获取人脸检测点
#     predictor_path = 'shape_predictor_81_face_landmarks.dat'
#     predictor = dlib.shape_predictor(predictor_path)
#     shape = predictor(img, d)
#     landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])

#     #裁切并保存
#     cropped_img = img[top:bottom, left:right, :]
#     return cropped_img, loc

def preprocess(depth_img, IR_img, aligned_img):
    try:
        cropped_aligned, location = crop(aligned_img)
    except TypeError:
        return None

    cropped_depth = depth_img[location[0]:location[1], location[2]:location[3], :]
    cropped_IR = IR_img[location[0]:location[1], location[2]:location[3], :]

    aligned_PR_mask = get_depth(cropped_aligned)
    aligned_bin_mask = get_mask(aligned_PR_mask)

    masked_depth = np.where(aligned_bin_mask[:,:,np.newaxis] == 0, 0, cropped_depth)
    masked_IR = np.where(aligned_bin_mask[:,:,np.newaxis] == 0, 0, cropped_IR)

    return masked_depth, masked_IR
