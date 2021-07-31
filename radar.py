import numpy as np
import cv2
import time
from math import *

# read the video
#cap = cv2.VideoCapture('HighwayIII_Original_sequence.avi')
cap = cv2.VideoCapture('bisaivideo2.mp4')
#cap = cv2.VideoCapture('test.avi')

fps = 0.0
temp = 10 # 用于跳过前 10 帧的
while True:
    t1 = time.time()
    ret, frame = cap.read()
    
    if temp > 0:  # 跳过前 10 帧噪声画面
        temp -= 1
        continue
    #cv2.imwrite("input.png", frame)
    #frame = cv2.resize(frame, (640,970), interpolation = cv2.INTER_AREA)  # 重新resize画面大小
    #cv2.imshow('input', frame)  # 原图
    
    # 测试用的视频画面裁剪，可删去
    #frame = frame[400:,:]
    #frameHeight = frame.shape[0]  # 画面帧高
    #frameWidth = frame.shape[1]  # 画面帧宽

    result, m_ = getPerson(frame, frameWidth, frameHeight)

    # 帧率计算(CPU)
    try :
        fps  = ( fps + (1./(time.time()-t1)) ) * 0.5
        #print("fps= %.2f"%(fps))
    except ZeroDivisionError as e:
        t1 = time.time()
        
    frame = cv2.putText(frame, "FPS= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow('result', result)
    k = cv2.waitKey(100)&0xff
    
    if k == 27:
        #cv2.imwrite("result.png", result) # 保存结束图片
        #cv2.imwrite("mask.png", m_)
        break
cap.release()
cv2.destroyAllWindows()