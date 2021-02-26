import time

import cv2
import numpy as np

from nets.retinaface_inference import Retinaface


if __name__ == '__main__':

    retinaface = Retinaface()

    #-------------------------------------#
    #   调用摄像头
    #   capture=cv2.VideoCapture("1.mp4")
    #-------------------------------------#
    capture = cv2.VideoCapture(0)

    fps = 0.0
    while True:
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        # 左右镜像对称
        frame = cv2.flip(frame, 1)
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 进行检测
        t2 = time.time()
        frame = np.array(retinaface.detect_image(frame))
        print('检测一帧视频需要{}ms'.format((time.time()-t2)*1000))
        # RGBtoBGR满足OpenCV显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % fps)
        frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break
