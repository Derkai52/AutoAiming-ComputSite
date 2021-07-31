#----------------------------------------------------#
#   对视频中的predict.py进行了修改，
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time
from datetime import datetime
import os
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

# net-work
import socket
import sys
import json
import math

# threading
import threading
import queue

# 网络通信建立
def netconnect():
    address = ('169.251.252.109', 5005)  # 服务端地址和端口
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(address)  # 尝试连接服务端
    except Exception:
        print('[!] Server not found ot not open')
        sys.exit()
    data = {"x": 0, "y": 0}
    shoot = 0
    connectmode = "network" # 设置连接模式为网络模式
    return s, data, shoot, connectmode

# 发送目标数据
def messmeg(frame, socket_s, data):
    # print(frame.shape) == (480, 640, 3)
    data['x'] = int((target_x - frame.shape[1] / 2) /1)
    data['y'] = int((target_y - frame.shape[0] / 2) /1)
    data['shoot'] = int(shoot)
    print('x',target_x,'y',target_y,'shoot',shoot)
    a_str = json.dumps(data)
    s.sendall(a_str.encode())
    data = s.recv(1024)  # 作为首发
    data = data.decode()

# 求两点间距离
def getDist_P2P(PointA, PointB):
    distance = math.pow((PointA[0] - PointB[0]), 2) + math.pow((PointA[1] - PointB[1]), 2)
    distance = math.sqrt(distance)
    return distance

# 图像显示
def outwindows(frame, outWinFlag):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGBtoBGR满足opencv显示格式
    if outWinFlag == "open":  # 全屏显示
        out_win = "output_style_full_screen"
        cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(out_win, frame)
    else:  # 仅按窗口实际大小显示
        cv2.imshow("video", frame)

# 读图线程
def threadshow(ClientAddress):
    global q # 图像缓冲区队列
    global producer # 线程实例化对象

    q = queue.Queue(maxsize = 2) # warning 一种投机取巧的方案。可能因缓存区存留导致拿到的不是最新一帧，但能保证是2帧范围内
    producer = Producer('Pro1.', q, ClientAddress)  # 调用对象，并传如参数线程名、实例化队列
    producer.start()  # 开启读图线程
    print('All threads terminate!')

#读图线程类
class Producer(threading.Thread):
    def __init__(self, t_name,queue,address):  # 传入线程名、实例化队列
        threading.Thread.__init__(self, name=t_name)  # t_name即是threadName
        self.data = queue
        self.address = address

    def run(self):
        global capture
        capture = cv2.VideoCapture(self.address) # bug 请使用变量控制 'http://169.251.252.109:8000/stream.mjpg'
        while True:
            ref, frame = capture.read()
            if self.data.full():
                self.data.get()
            self.data.put(frame)  # 图像写入队列


            #time.sleep(0.02)



select = int(input("\n1、网络推流模式 2、本地推流模式\n请选择调试模式："))
if select == 1:
    s, data, shoot, connectmode = netconnect() # 初始化网络通信  # warnning 载入模型需要时间。发送端可能出现接受请求后不断堆入缓冲区
    ClientAddress = input("\n请输入完整HTTP地址：\n例如：http://169.251.252.109:8000/stream.mjpg\n")
elif select == 2:
    connectmode = "local"  # 默认模式为本地调试模式

videoSaveFlag = int(input("\n1、开启录像 2、不开启录像\n请选择是否录像："))



if __name__ == "__main__":
    global q # 堆栈图像对象
    threadshow(ClientAddress) # 启动线程单独读图
    yolo = YOLO()

    #-------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测
    #   'video'表示视频检测
    #   'fps'表示测试fps
    #-------------------------------------------------------------------------#
    mode = "video"
    #-------------------------------------------------------------------------#
    #   connectmode用于指定读图的模式，“network”:网络读图模式 “local”:本地读图模式
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出才会完成完整的保存步骤，不可直接结束程序。
    #   outWinFlag 用于标识是否显示图像，仅为 "open" 时显示图像
    #-------------------------------------------------------------------------#
    if connectmode == "network":
        video_path = ClientAddress #读取网络推流
    else:
        connectmode == "local"
        video_path      = 0 # 读取本地推流

    if videoSaveFlag == 1: # 选择保存video
        video_save_path = 'video/'+datetime.now().strftime('%Y%m%d%H%M%S') + ".avi"
        video_fps       = 25.0
    else:video_save_path = "" # 不选择保存video

    outWinFlag = "open" # 显示图像

    if mode == "predict":
        '''
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_dr_txt.py，在get_dr_txt.py即实现了遍历还实现了目标信息的保存。
        2、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        3、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        4、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        5、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
                time.sleep(20)

    elif mode == "video":
        # capture=cv2.VideoCapture(video_path) # bug capture占用。需要用变量名代替
        global capture
        if video_save_path!="":
            if not os.path.exists('video'): # 自动建立video文件路径
                os.mkdir('video')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        #outflag = 100 # 前200帧不处理
        while(True):
            t1 = time.time()
            # 读取某一帧
            #ref,frame=capture.read()
            frame = q.get() # 从堆栈中读取
            if frame is None: # 读取帧为空
                print("Image is None!!!")
                continue

            # y,x  画面裁切
            #frame = frame[300:frame.shape[0] - 300, 400:frame.shape[1] - 400]

            ## 跳过前 n 帧
            # if outflag >= 0:
            #     outflag-=1
            #     cv2.imshow("outshow", frame)
            #     c = cv2.waitKey(1) & 0xff
            #     continue

            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            # 进行检测
            # frame= np.array(yolo.detect_image(frame))
            frame, target_x, target_y, shoot_flag, dist = yolo.detect_image(frame) # bug 传出参数过多。可优化
            frame = np.array(frame)

            # 获取图像初始信息
            img_w = frame.shape[1]  # 获取图像尺寸的宽
            img_h = frame.shape[0]  # 获取图像尺寸的高
            img_2w = int(img_w / 2) # 获取图像尺寸半宽
            img_2h = int(img_h / 2) # 获取图像尺寸半高
            cv2.line(frame, (img_2w, 0), (img_2w, img_h), (123, 231, 0), 2)
            cv2.line(frame, (0, img_2h), (img_w, img_2h), (123, 231, 0), 2)

            if connectmode == "local": # bug 为防止下面变量未定义报错。变量初始化刷新可在外面执行
                target_x, target_y = 0,0 # 目标相对坐标
                shoot_flag = 0 # 开火标识
                dist = 0 # 平面距离
            shoot = 0  # bug 可在外面刷新

            # 获取目标平面距离
            realdist = getDist_P2P((img_2w,img_2h),(target_x,target_y))

            if shoot_flag == 1 and realdist <= dist: # 设置开火距离阈值(1、可开火标识  2、达到开火阈值)
                cv2.line(frame, (img_2w, img_2h), (target_x, target_y), (255, 50, 0), 2)  # 开火线(红线)
                shoot = 1
            else:
                cv2.line(frame, (img_2w, img_2h), (target_x, target_y), (255, 231, 0), 2)  # 目标远(黄线)
                shoot = 0

            # 发送目标数据
            if connectmode == "network":
                messmeg(frame, s, data)

            # FPS帧率显示
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            #print("FPS= %.2f"%(fps))
            frame = cv2.putText(frame, "FPS= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # 显示FPS帧率
            cv2.circle(frame, (target_x, target_y), 4, (0, 0, 255), 4)  # 显示枪口中心

            # 显示图像
            outwindows(frame, outWinFlag)

            c= cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            # 按键操作
            if c==27: # ESC键
                capture.release()
                break

            if c==115: # s键 暂停图像输入
                while True:
                    frame = cv2.putText(frame, "Press any key to continue", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("video", frame)
                    c = cv2.waitKey(1) & 0xff
                    if c == 115:
                        break
                continue

        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 100
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        #print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    else:raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")
    s.close() # 关闭连接