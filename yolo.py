import colorsys
import os
import time

import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from nets.yolo4_tiny import yolo_body, yolo_eval
from utils.utils import letterbox_image


# hhhh
# 防止显存爆炸，限制GPU使用
import tensorflow as tf
import cv2
import math
#import numpy as np
import keras

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# 求两点间距离
def getDist_P2P(PointA, PointB):
    distance = math.pow((PointA[0] - PointB[0]), 2) + math.pow((PointA[1] - PointB[1]), 2)
    distance = math.sqrt(distance)
    return distance
##

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和phi都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path、classes_path和phi参数的修改
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/yolov4_tiny_weights_coco.h5',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/coco_classes.txt',
        #-------------------------------#
        #   所使用的注意力机制的类型
        #   phi = 0为不使用注意力机制
        #   phi = 1为SE
        #   phi = 2为CBAM
        #   phi = 3为ECA
        #-------------------------------#
        "phi"               : 0,  
        "score"             : 0.2,
        "iou"               : 0.2,
        "max_boxes"         : 100,  
        #-------------------------------#
        #   显存比较小可以使用416x416
        #   显存比较大可以使用608x608
        #-------------------------------#
        "model_image_size"  : (416, 416),
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #---------------------------------------------------#
        #   计算先验框的数量和种类的数量
        #---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        #---------------------------------------------------------#
        #   载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        #   否则先构建模型再载入
        #---------------------------------------------------------#
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes, self.phi)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2, ))

        #---------------------------------------------------------#
        #   在yolo_eval函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        #---------------------------------------------------------#
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                num_classes, self.input_image_shape, max_boxes = self.max_boxes,
                score_threshold = self.score, iou_threshold = self.iou, letterbox_image = self.letterbox_image)
        return boxes, scores, classes

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #---------------------------------------------------------#

        image = image.convert('RGB')

        #image = image[200:image.shape[0] - 100, 100:image.shape[1] - 100]
        # image = image.convert('RGB')
        # frame = image.convert('RGB')
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)

        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #---------------------------------------------------------#
        #   设置字体
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        #---------------------------------------------------------#
        #   目标信息初始化
        #---------------------------------------------------------#
        target_x = int(image.size[0]/2) # init site
        target_y = int(image.size[1]/2)
        shoot_flag = 0
        dist = 0
        distlist = [] # 距离列表
        targetlist = [] #目标列表
        rangelist = [] # 目标框列表

        #---------------------------------------------------------#
        #   对检出目标进行条件筛选，获得最佳射击目标
        #---------------------------------------------------------#
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            if predicted_class == 'person': #'car'or predicted_class == 'truck':  # 可对检出的类单独操作
                shoot_flag = 1 # 当识别到人会给1
                box = out_boxes[i]
                score = out_scores[i]

                top, left, bottom, right = box
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5

                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                # 画框框
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                print(label, top, left, bottom, right)
                rangewight = right - left # 框宽
                rangeheight = bottom - top # 框高

                if rangewight - rangeheight >= 0:  # 宽 - 高 >= 0  筛选不符合条件的(已经阵亡倒地的目标)
                    continue

                target_x = int(rangewight/2+left)
                target_y = int(rangeheight/2+top - rangeheight*0.3)  # 锁头(头部占身体上半部分约20%)

                # 最大开火距离(平面)
                dist = int(rangewight/2) # bug 默认是以宽度最小为阈值
                #print("X:",target_x,"Y:",target_y)

                # 自适应显示
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])

                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
                # bug 应当是给一个自适应的。但是image变量和前面有冲突，需要单独获取原图像中心点
                img_w = 960#image.size[0]  # 获取图像尺寸的宽
                img_h = 540#image.size[1]  # 获取图像尺寸的高
                img_2w = int(img_w / 2)  # 获取图像尺寸半宽
                img_2h = int(img_h / 2)  # 获取图像尺寸半高
                distlist.append(getDist_P2P((img_2w, img_2h), (target_x, target_y))) #
                targetlist.append([target_x,target_y])     # [x,y]
                rangelist.append([right-left,bottom-top])  # [宽,高]

        # 获取当前帧平面距离最近的目标
        if len(distlist) != 0:
            MinDistIndex = distlist.index(min(distlist))
            target_x, target_y = targetlist[MinDistIndex][0],targetlist[MinDistIndex][1]

        return image, target_x, target_y, shoot_flag, dist

    def get_FPS(self, image, test_interval):
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0})
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def close_session(self):
        self.sess.close()
