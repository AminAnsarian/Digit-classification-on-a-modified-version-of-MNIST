#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:09:53 2019

@author: amin
"""
import pandas as pd
import cv2 
import matplotlib.pyplot as plt

def model_submit(y_pred):
    c = ['Id','Category']
    sample_sub = pd.read_csv('sample_submission.csv')
    values = sample_sub.values
    values[:,1] = y_pred
    df = pd.DataFrame(values)
    df.to_csv("submission.csv", header=c,index=False)
    
def threshs(img,thr):
    ret,thresh1 = cv2.threshold(img,thr,255,cv2.THRESH_BINARY)
    return thresh1
def disp(img):
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
