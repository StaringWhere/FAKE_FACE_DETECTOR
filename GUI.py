#import sys
import tkinter as tk
import pyrealsense2 as rs
import numpy as np
import cv2
from preprocess import preprocess
from inference import check
import threading as th
from PIL import Image, ImageTk

'''
视频流配置
'''
# 创建一个pipeline
pipeline = rs.pipeline()

# 配置pipeline
config = rs.config()
from_bag = 1
if from_bag:
    # 从bag文件获取流
    # config.enable_device_from_file('20200503_111200.bag')
    # config.enable_device_from_file('videos/real.bag') # real 0.95 real 0.91
    # config.enable_device_from_file('videos/eye_flat.bag') # fake 0.87 fake 0.89
    # config.enable_device_from_file('videos/eye_nose_flat.bag') # fake 0.88 fake 0.97
    # config.enable_device_from_file('videos/eye_mouth_flat.bag') # fake 0.78 fake 0.96
    # config.enable_device_from_file('videos/eye_nose_mouth_flat.bag') # fake 0.91 fake 0.95
    # config.enable_device_from_file('videos/eye_curved.bag') # fake 0.95 fake 0.83
    # config.enable_device_from_file('videos/eye_nose_curved.bag') # fake 0.94 fake 0.94
    # config.enable_device_from_file('videos/eye_mouth_curved.bag') # fake 0.96 fake 0.92
    config.enable_device_from_file('videos/eye_nose_mouth_curved.bag') # fake 0.95 fake 0.96
else:
    # 从摄像头获取流
    cam_width, cam_height, cam_fps = 640, 480, 0
    config.enable_stream(rs.stream.depth, cam_width,
                        cam_height, rs.format.z16, cam_fps)
    config.enable_stream(rs.stream.color, cam_width,
                        cam_height, rs.format.bgr8, cam_fps)
    config.enable_stream(rs.stream.infrared, cam_width,
                        cam_height, rs.format.y8, cam_fps)

# 开始收集数据
profile = pipeline.start(config)

# 获取bag的宽度和高度
if from_bag:
    cam_width = profile.get_streams()[0].as_video_stream_profile().width()
    cam_height = profile.get_streams()[0].as_video_stream_profile().height()

# 设置Emitter Enabled
if not from_bag:
    target = profile.get_device().first_depth_sensor().set_option(rs.option.emitter_enabled, 2)

# 获取深度传感器放大倍数
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 深度裁切阈值
clipping_distance_in_meters = 1
clipping_distance = clipping_distance_in_meters / depth_scale

# 创建align对象
# 令其他图像都对齐到depth图像上
align_to = rs.stream.depth
align = rs.align(align_to)

count = 0
ispause = 0
# 用于推理判断的图片
inf_images = None

# 扫描速度
v = 2
# 初始y坐标
y = 0
# 初始方向：1表示向下,-1表示向上
turn = 1

# 是否结束
isFinish = 0
def bgrun(depth_img, ir_img, aligned_img):
    global isFinish
    text.set('WORKING...')
    try:
        masked_depth, masked_ir = preprocess(depth_img, ir_img, aligned_img)
        masked_depth = Image.fromarray(masked_depth)
        masked_ir = Image.fromarray(masked_ir)
        res = check(masked_depth, masked_ir)
        isFinish = 1
        if(res==0):
            result.config(fg='red')
            text.set('FAKE')
        else:
            result.config(fg='green')
            text.set('REAL')
    except TypeError:
        result.config(fg='yellow')
        text.set('FACE NOT DETECTED')

def play():
    global pipeline,count,inference,inf_images,ispause,y,turn,isFinish
    # 判断是否暂停
    if ispause:
        return
    # 获取帧
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    infrared_frame = frames.get_infrared_frame()

    # 对齐帧
    aligned_frames = align.process(frames)
    aligned_color_frame = aligned_frames.get_color_frame()
    aligned_infrared_frame = aligned_frames.get_infrared_frame()
    
    # 验证帧是否有效
    if not (color_frame and depth_frame and infrared_frame and aligned_color_frame and aligned_infrared_frame):
        return
    
    count = count + 1
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    infrared_image = np.asanyarray(infrared_frame.get_data())
    aligned_color_image = np.asanyarray(aligned_color_frame.get_data())
    aligned_infrared_image = np.asanyarray(aligned_infrared_frame.get_data())

    color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_resized = cv2.resize(color_rgb,(400, 320))
    # 添加扫描线
    if(isFinish==0):
        if y==0:
            turn = 1
        elif y==320:
            turn = -1
        y = y + turn*v
        color_resized = cv2.line(color_resized, (0,int(y)), (400,int(y)), (0,180,0), 2)
    # 绘制在界面上
    colortk = ImageTk.PhotoImage(image=Image.fromarray(color_resized))
    main_panel.imgtk = colortk
    main_panel.config(image=colortk)

    # 用realsense上色depth
    colorizer = rs.colorizer()
    # 设置visual_preset: 0 - dynamic 1 - fixed
    colorizer.set_option(rs.option.visual_preset, 0)
    # 设置colormap，会被visual_present刷新，所以要放在visual_present后面
    # 0 - Jet 1 - Classic 2 - WhiteToBlack 3 - BlackToWhite 4 - Bio 5 - Cold 6 - Warm 7 - Quantized 8 - Pattern
    colorizer.set_option(rs.option.color_scheme, 2)
    # 设置min_distance
    colorizer.set_option(rs.option.min_distance, 0.3)
    # 设置max_distance
    colorizer.set_option(rs.option.max_distance, 1.0)
    depth_image_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())  

    # 统一维度
    infrared_image_3d = np.dstack((infrared_image, infrared_image, infrared_image))
    aligned_infrared_image_3d = np.dstack((aligned_infrared_image, aligned_infrared_image, aligned_infrared_image))

    # 保存要处理的帧
    if count == 30:
        t= th.Thread(target=bgrun,args=(depth_image_colormap, infrared_image_3d, aligned_color_image))#创建线程
        t.setDaemon(True)# 设置为后台线程，这里默认是False，设置为True之后则主线程不用等待子线程
        t.start()
    #只测试一次
    if count==60:
        count = 31

    aligned_color_rgb = cv2.cvtColor(aligned_color_image, cv2.COLOR_BGR2RGB)
    depth_image_colormap = cv2.putText(depth_image_colormap,'Depth',(0,40),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
    infrared_image_3d = cv2.putText(infrared_image_3d,'IR',(0,40),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
    aligned_color_image = cv2.putText(aligned_color_rgb,'Aligned',(0,40),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
    aligned_infrared_image_3d = cv2.putText(aligned_infrared_image_3d,'Aligned_IR',(0,40),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)

    # 渲染
    images1 = np.hstack((depth_image_colormap, infrared_image_3d))
    images2 = np.hstack((aligned_color_image, aligned_infrared_image_3d))
    images = np.vstack((images1, images2))

    images_resized = cv2.resize(images,(640, 480))
    imgstk = ImageTk.PhotoImage(image=Image.fromarray(images_resized))
    
    minor_panel.imgstk = imgstk
    minor_panel.config(image=imgstk)
    window.after(1, play)
    
# 开始
def onStart():
    content.config(state='normal')
    content.insert('end', 'Loading video stream from camera...\n')
    content.config(state='disabled')
    play()
    start.config(state='disabled')

# 暂停
def onPause():
    content.config(state='normal')
    content.insert('end', 'Pause video stream\n')
    content.config(state='disabled')
    global ispause
    ispause = 1

# 继续
def onProceed():
    content.config(state='normal')
    content.insert('end', 'Continue video stream\n')
    content.config(state='disabled')
    global ispause,count,isFinish
    count = 0
    isFinish = 0
    result.config(fg='black')
    text.set('')

    if ispause == 1:
        ispause = 0
        play()

# 创建窗口
window = tk.Tk()
window.title('FAKE FACE DETECTOR')
window.minsize(1400, 800)
window.maxsize(1400, 800)

# GUI界面设计
subWin1 = tk.Frame(window, width=700, height=500)
subWin2 = tk.Frame(window, width=700, height=500)
subWin3 = tk.Frame(window, width=1400, height=300)
subWin1.grid(row=0, column=0)
subWin2.grid(row=0, column=1)
subWin3.grid(row=1, columnspan=2)
main_view = tk.Label(subWin1, text='M\na\ni\nn\n\nV\ni\ne\nw\n', font=('Helvetica', 25))
minor_view = tk.Label(subWin1, text='M\ni\nn\no\nr\n\nV\ni\ne\nw\n', font=('Helvetica', 25))
main_view.place(x=70, y=40, width=30, height=300)
minor_view.place(x=650, y=40, width=30, height=300)
main_panel = tk.Label(subWin1, relief='ridge')
main_panel.place(x=150, y=20, width=400, height=320)
minor_panel = tk.Label(subWin2, relief='ridge')  
minor_panel.place(x=30, y=10, width=640, height=480)

text = tk.StringVar()
result = tk.Label(subWin1, textvariable=text, font=('Times', 25, 'bold'), relief='sunken')
result.place(x=240, y=370, width=200, height=50)
start = tk.Button(subWin1, text="start", font=('Helvetica', 20), command=onStart,  relief='raised')
pause = tk.Button(subWin1, text="pause", font=('Helvetica', 20), command=onPause,  relief='raised')
proceed = tk.Button(subWin1, text="proceed", font=('Helvetica', 20), command=onProceed, relief='raised')
start.place(x=120, y=440)
pause.place(x=300,y=440)
proceed.place(x=480,y=440)

content = tk.Text(subWin3, state='disabled', font=('Helvetica', 20))
content.place(x=20, y=10, height=280, width=1340)
scroll = tk.Scrollbar(subWin3)
scroll.place(x=1360, y=10, height=280, width=20)
scroll.config(command=content.yview)
content.config(yscrollcommand=scroll.set)

# 窗口主循环
window.mainloop()
