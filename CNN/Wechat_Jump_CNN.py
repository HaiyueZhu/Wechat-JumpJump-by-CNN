# coding: utf-8

"""
Using Convolutional Neural Network (CNN) to play Wechat Jump-Jump Game:
    The idea to using CNN to play Wechat Jump-Jump Game is to mimic our human how the play this game that we see the start
    point and the target point, and then judge their distance to press the scree. So we can train a CNN model to find the
    start point's and the target point's position in an image, and then calculate their distance to press using ADB tool.

    For training the CNN model, there are some image processing based method to play Wechat Jump-Jump Game, as we found the
    following one: https://github.com/wangshub/wechat_jump_game
    We used their method to generate the training images by auto run it and saving every step's image with the name as the
    four coordinates of the starting jump point and ending point (target).

    There are three steps for playing the games:
      1, Run GetCNNTrainingImages.py to get the CNN training images
      2, Run CNN_Training.py to creat a CNN (tensorflow) and train it
      3, Play the jump-jump game using the trained CNN
"""

import os
import sys
import subprocess
import time
import math
from PIL import Image
import random
from six.moves import input
import cv2
import tensorflow as tf
import numpy as np
import CNN_Training

try:
    from common import debug, config
except ImportError:
    print('Please including the folder "common" and "config"!!')
    exit(-1)


debug_switch = False    # debug 开关，需要调试的时候请改为：True
config = config.open_accordant_config()

# Magic Number，不设置可能无法正常执行，请根据具体截图从上到下按需设置，设置保存在 config 文件夹中
under_game_score_y = config['under_game_score_y']
press_coefficient = config['press_coefficient']       # 长按的时间系数，请自己根据实际情况调节
piece_base_height_1_2 = config['piece_base_height_1_2']   # 二分之一的棋子底座高度，可能要调节
piece_body_width = config['piece_body_width']             # 棋子的宽度，比截图中量到的稍微大一点比较安全，可能要调节


screenshot_way = 2


def pull_screenshot():
    '''
    新的方法请根据效率及适用性由高到低排序
    '''
    global screenshot_way
    if screenshot_way == 2 or screenshot_way == 1:
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        screenshot = process.stdout.read()
        if screenshot_way == 2:
            binary_screenshot = screenshot.replace(b'\r\n', b'\n')
        else:
            binary_screenshot = screenshot.replace(b'\r\r\n', b'\n')
        f = open('autojump.png', 'wb')
        f.write(binary_screenshot)
        f.close()
    elif screenshot_way == 0:
        os.system('adb shell screencap -p /sdcard/autojump.png')
        os.system('adb pull /sdcard/autojump.png .')


def set_button_position(im):
    '''
    将 swipe 设置为 `再来一局` 按钮的位置
    '''
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    w, h = im.size
    left = int(w / 2)
    top = int(1584 * (h / 1920.0))
    left = int(random.uniform(left-50, left+50))
    top = int(random.uniform(top-10, top+10))    # 随机防 ban
    swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, left, top


def jump(distance):
    '''
    跳跃一定的距离
    '''
    press_time = distance * press_coefficient
    press_time = max(press_time, 200)   # 设置 200ms 是最小的按压时间
    press_time = int(press_time)
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
        x1=swipe_x1,
        y1=swipe_y1,
        x2=swipe_x2,
        y2=swipe_y2,
        duration=press_time
    )
    os.system(cmd)
    return press_time


def check_screenshot():
    '''
    检查获取截图的方式
    '''
    global screenshot_way
    if os.path.isfile('autojump.png'):
        os.remove('autojump.png')
    if (screenshot_way < 0):
        print('暂不支持当前设备')
        sys.exit()
    pull_screenshot()
    try:
        Image.open('./autojump.png').load()
    except Exception:
        screenshot_way -= 1
        check_screenshot()
        print('screenshot unsuccessful!')



def yes_or_no(prompt, true_value='y', false_value='n', default=True):
    default_value = true_value if default else false_value
    prompt = '%s %s/%s [%s]: ' % (prompt, true_value, false_value, default_value)
    i = input(prompt)
    if not i:
        return default
    while True:
        if i == true_value:
            return True
        elif i == false_value:
            return False
        prompt = 'Please input %s or %s: ' % (true_value, false_value)
        i = input(prompt)

def main():

    op = yes_or_no('Enter "y" to start if ensure ADB is installed and Jump-Jump is open:')
    if not op:
        print('bye')
        return
    debug.dump_device_info()
    check_screenshot()


    x = tf.placeholder(tf.float32, [None, 51, 86])
    y_conv, keep_prob = CNN_Training.deepnn(x)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('./TF_Saved_Model/'))

    Num = 0
    while(True):
        pull_screenshot()

        im_cv = cv2.imread('./autojump.png')
        im_cvrs = cv2.resize(im_cv, None, fx=ReshapeRatio1, fy=ReshapeRatio1, interpolation=cv2.INTER_CUBIC)
        im_cvrs_back = cv2.resize(im_cv, None, fx=ReshapeRatio1, fy=ReshapeRatio1, interpolation=cv2.INTER_CUBIC)

        H_org, W_org, _ = im_cv.shape
        H_rs_org, W_rs_org, _ = im_cvrs.shape

        im_cvrs = im_cvrs[250:506,:]

        im_cvrs_cnn = cv2.resize(im_cvrs, None, fx=ReshapeRatio2, fy=ReshapeRatio2, interpolation=cv2.INTER_CUBIC)
        im_cvrs_cnn_gray = cv2.cvtColor(im_cvrs_cnn, cv2.COLOR_BGR2GRAY)
        cv2.imshow('im_cvrs_cnn_gray', im_cvrs_cnn_gray)


        im_cvrs_cnn_gray = (im_cvrs_cnn_gray-127)/255
        x_batch = im_cvrs_cnn_gray[np.newaxis,:,:]
        prediction = sess.run([y_conv],feed_dict={x: x_batch, keep_prob: 1})
        prediction = prediction[0][0]

        piece_x = int(prediction[0]*W_org)
        piece_y = int(prediction[1]*H_org)
        board_x = int(prediction[2]*W_org)
        board_y = int(prediction[3]*H_org)
        print('CNN Predicted Position:\n', piece_x,piece_y,board_x,board_y)


        piece_x_rs = int(prediction[0]*W_rs_org)
        piece_y_rs = int(prediction[1]*H_rs_org)
        board_x_rs = int(prediction[2]*W_rs_org)
        board_y_rs = int(prediction[3]*H_rs_org)

        cv2.circle(im_cvrs_back, (piece_x_rs, piece_y_rs), 15, (255, 0, 0), -1)
        cv2.circle(im_cvrs_back, (board_x_rs, board_y_rs), 15, (0, 0, 255), -1)

        cv2.imshow('Image', im_cvrs_back)
        cv2.waitKey(500)

        ts = int(time.time())
        im = Image.open('./autojump.png')
        set_button_position(im)


        jump(math.sqrt((board_x - piece_x) ** 2 + (board_y - piece_y) ** 2))
        if debug_switch:
            debug.save_debug_screenshot(ts, im, piece_x, piece_y, board_x, board_y)
            debug.backup_screenshot(ts)
        time.sleep(random.uniform(1.9, 2.2))   # 为了保证截图的时候应落稳了，多延迟一会儿，随机值防 ban


if __name__ == '__main__':
    ReshapeRatio1 = 0.4
    ReshapeRatio2 = 0.2
    main()
