import tensorflow as tf
import os
import numpy as np
import cv2


from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


yolov3_loss_op_module = tf.load_op_library('../new_op.so')

@ops.RegisterGradient("YoloLoss")
def _yolo_loss_grad(op, grad):
    input_0_grad = grad
    input_1_grad = tf.zeros_like(op.inputs[1])
    return [input_0_grad, input_1_grad]


def yolov3_loss(feature_map, ground_truth, mask, anchors, img_height, img_width, ignore_thresh = 0.5, is_training = True, formate = "NHWC"):
    '''
        feature_map: NHWC
        ground_truth: x,y,w,h,id, (x,y)is image center coord
        mask: 1-D 3 element,[6,7,8]
        anchors: 1-D all anchors coord
        img_height: input img height
        img_width: input img width
        ignore_thresh: default is 0.5
    '''
    if formate == "NHWC":
        feature_map = tf.transpose(feature_map, [0, 3, 1, 2])
    delta = yolov3_loss_op_module.yolo_loss(feature_map, ground_truth, mask = mask, anchors = anchors, img_height = img_height,
                                             img_width = img_width, ignore_thresh = ignore_thresh, is_training = is_training)
    square_x = tf.square(delta)
    loss = tf.reduce_sum(square_x)
    return loss, delta


if __name__ == '__main__':



    img_w = 416
    img_h = 416
    batch = 1
    anchor_num = 3
    total = 9
    classes = 2
    output_channel = anchor_num*(classes + 4 + 1)

    h = 52
    w = 52

    feature_map = []
    ground_truth = []
    mask = []
    delta = []
    ff = open('./test_data/feature_map', "r")
    lines = ff.readlines()
    for line in lines:
        line = line.strip()
        feature_map.append(float(line))
    ff.close()

    ff = open('./test_data/ground_truth', "r")
    lines = ff.readlines()
    for line in lines:
        line = line.strip()
        ground_truth.append(float(line))
    ff.close()

    ff = open('./test_data/mask', "r")
    lines = ff.readlines()
    for line in lines:
        line = line.strip()
        mask.append(int(line))
    ff.close()

    ff = open('./test_data/delta', "r")
    lines = ff.readlines()
    for line in lines:
        line = line.strip()
        delta.append(float(line))
    ff.close()

    feature_map = np.array(feature_map)
    feature_map = np.reshape(feature_map, ( batch,output_channel,h,w ) )
#     mask = np.array(mask)
    #x,y,w,h,id
    ground_truth = np.array(ground_truth)
    ground_truth = ground_truth.astype(np.float32)

    # feature_map = np.random.randint(10,100,size = (batch,h,w,output_channel))
    # feature_map = feature_map / 100.0
    #
    # mask = np.array([6,7,8])
    #
    # #x,y,w,h,id
    # ground_truth = np.array([0.2,0.3,0.4,0.5,0.0])
    # ground_truth = ground_truth.astype(np.float32)

    anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]

    feature_map = tf.convert_to_tensor(feature_map, dtype=tf.float32)
#     with tf.device("/cpu:0"):
#         mask = tf.convert_to_tensor(mask, dtype=tf.int32)
#         print(mask)
    ground_truth = tf.convert_to_tensor(ground_truth, dtype=tf.float32)
    #NHWC -> NCHW
    # feature_map = tf.transpose(feature_map, [0, 3, 1, 2])
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        loss, _ = yolov3_loss(feature_map, ground_truth, mask = mask, anchors = anchors,
                               img_height = img_h, img_width = img_w, ignore_thresh = 0.5, is_training = False, formate = "NCHW")
        res, boxes = (sess.run([loss, _]))
#         g = tf.gradients(loss,feature_map)
#         res, g_ = (sess.run([loss, g]))
        #nboxes: 7nboxes: 20nboxes: 172face: 98%
        print("python end!!!")
        print(res)
        img = cv2.imread("/home/dzhang/work/dzhang/darknet/test_img/60e8f0416e8c2620a1cf5264075d1a60.jpg")
        boxes = np.array(boxes)
        boxes = boxes.flatten().tolist()[:1032]
        boxes = np.array(boxes)
        boxes = np.reshape(boxes,(boxes.size / 6, 6))
        boxes = boxes.tolist()
        for tmp in boxes:
            tmp = [float(tem) for tem in tmp]
            is_break = False
            for tem in tmp:
                if tem < 0:
                    is_break = True
            if is_break:
                continue
            if tmp[4] <0.5 and tmp[5] < 0.5:
                continue
            up_left = (int(tmp[0]),int(tmp[2]))
            down_right = (int(tmp[1]),int(tmp[3]))
            cv2.rectangle(img,up_left,down_right,(0, 255, 0), 2)
        cv2.imshow("tt",img)
        cv2.waitKey(0)
#         res = res * 1e6
#         res = res.astype(np.int32)
#         res = res / 1e6
#         res = res.flatten().tolist()
#         a = np.asarray(res) - np.asarray(delta)
#         a = np.sum(a)
#         print(a)
#         for tmp in range(len(res)):
#             if res[tmp] != delta[tmp]:
#                 print(tmp)
#                 print("%f, %f" % (res[tmp],delta[tmp]))
#         print(g_)

