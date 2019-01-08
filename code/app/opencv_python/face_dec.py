# -*- coding: UTF-8 -*-

import numpy as np 
import cv2 
import sys
import time 
import argparse


def SaveFile():
#cap = cv2.VideoCapture(0) 

  cap = cv2.VideoCapture("rtsp://admin:ABC_123456@172.17.208.150:554/Streaming/Channels/101?transportmode=unicast")
  cur_time = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))#time.strftime('%Y.%m.%d',time.localtime(time.time()))
  print(cur_time)
# Define the codec and create VideoWriter object 
  fourcc = cv2.VideoWriter_fourcc(*'XVID') 
#fourcc = cv2.VideoWriter_fourcc()
  file_name = cur_time + '.avi'
  #out = cv2.VideoWriter('output3.avi',fourcc, 20.0, (1920,1080)) 
  out = cv2.VideoWriter(file_name,fourcc, 20.0, (1920,1080)) 

  while(cap.isOpened()): 
    ret, frame = cap.read() 
    if ret==True: 
      out.write(frame)
    #frame = cv2.flip(frame,1) # write the flipped frame out.write(frame) 
    # write the flipped frame
    
      cv2.imshow('frame',frame) 
      if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
    else: 
      break 
# Release everything if job is finished 
  cap.release() 
  out.release() 
  cv2.destroyAllWindows()


def CatVideo():
    cv2.namedWindow("CaptureFace")
    #1调用摄像头,也可以读取工作目录下的视频'/*/*/*.avi'
    cap=cv2.VideoCapture("rtsp://admin:ABC_123456@172.17.208.150:554/Streaming/Channels/101?transportmode=unicast")
    #2人脸识别器分类器  GIT上面有开源的分类集，可以从下面的云盘里下载
    classfier=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    color=(0,255,0)
    while cap.isOpened():
        ok,frame=cap.read()
        if not ok:
            break
        #3灰度转换
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #4人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:            #大于0则检测到人脸                                   
            for faceRect in faceRects:  #单独框出每一张人脸
                 x, y, w, h = faceRect  #5画图   
                 cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)
        cv2.imshow("CaptureFace",frame)
        if cv2.waitKey(10)&0xFF==ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 


def FaceDec():
#cap = cv2.VideoCapture(0) 

  cap = cv2.VideoCapture("rtsp://admin:ABC_123456@172.17.208.150:554/Streaming/Channels/101?transportmode=unicast")

  classfier=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
  color=(0,255,0)

  while(cap.isOpened()): 
    ret, frame = cap.read() 
    if ret==True: 
      grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
      if len(faceRects) > 0:
        for faceRect in faceRects:
          x, y, w, h = faceRect  #5画图   
          cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)
    
      cv2.imshow('faceDec',frame) 
      if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
    else: 
      break 
# Release everything if job is finished 

  cap.release() 

  cv2.destroyAllWindows()


def face_detector():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args = vars(ap.parse_args())
 
# 如果video参数为None，那么我们从摄像头读取数据
    if args.get("video", None) is None:
        cap = cv2.VideoCapture("rtsp://admin:ABC_123456@172.17.208.150:554/Streaming/Channels/101?transportmode=unicast")
        time.sleep(0.25)
 
# 否则我们读取一个视频文件
    else:
        cap = cv2.VideoCapture(args["video"])

    classfier=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    color=(0,255,0)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
          grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
          faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
          if len(faceRects) > 0:
            for faceRect in faceRects:
              x, y, w, h = faceRect  #5画图   
              cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)
    
          cv2.imshow('faceDec',frame) 
          if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
        else: 
          break 
# Release everything if job is finished 

    cap.release() 

    cv2.destroyAllWindows()

SaveFile()
#CatVideo()
#FaceDec()
#face_detector()
