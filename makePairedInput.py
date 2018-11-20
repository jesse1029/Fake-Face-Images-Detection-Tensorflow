
import os
import cv2
import numpy as np
import random as rn
from Queue import Queue
import threading
import time
import signal
import sys
import pdb


fn='results'
fn2 = fn+'train.txt'

exclusive_list = ['celeba_wgan_gp', 'progressGAN', 'celeba_wgan', 'celeba_dcgan', 'celeba_lsgan']

split_num = len(exclusive_list)

for ext_num in range(split_num):

    f = open('train%s.txt'%(exclusive_list[ext_num]),'r')
    flist = f.readlines()

    fake_content = []
    real_content = []
    for i in range(len(flist)):
        fn1, lab1=flist[i].split(' ')
        if int(lab1)==1:
            real_content.append(fn1)
        else:
            fake_content.append(fn1)


    glen1=len(real_content)
    flen1=len(fake_content)
    seq1 = range(0,glen1)
    rn.shuffle(seq1)
    seq12 = range(0,glen1)
    rn.shuffle(seq12)


    seq2 = range(0,flen1)
    rn.shuffle(seq2)
    seq22 = range(0,flen1)
    rn.shuffle(seq22)

    print('Found %d fake images and %d real images'%(flen1, glen1))

    targetLen = 1000000
    balanceFactor = 0.5
    posLen = int(targetLen*balanceFactor)
    pcnt = 0
    cnt = 0

    text_file = open("pairwise_%s.txt"%(exclusive_list[ext_num]), "w")

    for i in range(posLen):
        if rn.random()>0.5:
            text_file.write('%s %s 1\n'%(real_content[seq1[i % glen1]], real_content[seq12[i % glen1]]))
        else:
            text_file.write('%s %s 1\n'%(fake_content[seq2[i % flen1]], fake_content[seq22[i % flen1]]))
        if i%10000==0:
            print("The positive sampling progress is " + str(i  * 100.0 / posLen)  + "%")
        if i>0 and ((i%glen1)==0):
            rn.shuffle(seq1)
            rn.shuffle(seq12)
            print('Shuffle real sequences')
        if i>0 and ((i%flen1)==0):
            rn.shuffle(seq2)
            rn.shuffle(seq22)
            print('Shuffle fake sequences at i=%d'%(i))

    for i in range(targetLen-posLen):
        if rn.random()>0.5:
            text_file.write('%s %s 0\n'%(real_content[seq1[i % glen1]], fake_content[seq22[i % flen1]]))
        else:
            text_file.write('%s %s 0\n'%(fake_content[seq2[i % flen1]], real_content[seq12[i % glen1]]))
        if i%10000==0:
            print("The negative sampling progress is " + str(i  * 100.0 / posLen)  + "%")
        if i>0 and (i%glen1==0):
            rn.shuffle(seq1)
            rn.shuffle(seq12)
        if i>0 and (i%flen1==0):
            rn.shuffle(seq2)
            rn.shuffle(seq22)

    text_file.close()
