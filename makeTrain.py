
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


fn='data/celebA'
myfiles = os.listdir(fn)
glen1 = len(myfiles)
real_content = []

for i in range(glen1):
	real_content.append('%s/%s' % (fn,  myfiles[i]))

fn='results'
subdirs = os.listdir(fn)
dir_num = len(subdirs)
print(subdirs)
exclusive_list = ['celeba_wgan_gp', 'progressGAN', 'celeba_wgan', 'celeba_dcgan', 'celeba_lsgan']

split_num = len(exclusive_list)

for ext_num in range(split_num):
    
    fake_content = []
    for i in range(dir_num):
        
        if exclusive_list[ext_num] != subdirs[i]:
            print("Processing dir %s"%(subdirs[i]))
            myfiles = os.listdir('%s/%s/' % (fn, subdirs[i]))
            files_num = len(myfiles)
            for j in range(files_num):
                fake_content.append('%s/%s/%s' % (fn, subdirs[i], myfiles[j]))
    flen1 = len(fake_content)

    seq1 = range(0,glen1)
    rn.shuffle(seq1)
    seq2 = range(0,flen1)
    rn.shuffle(seq2)


    print('Found %d fake images and %d real images'%(flen1, glen1))

    targetLen = glen1 - 5000
    pcnt = 0
    cnt = 0

    text_file = open("train%s.txt"%(exclusive_list[ext_num]), "w")
    for i in range(targetLen):
        text_file.write('%s 1\n'%(real_content[seq1[i % glen1]]))
        text_file.write('%s 0\n'%(fake_content[seq2[i % flen1]]))
    text_file.close()

    fake_content = []
    print("Processing dir %s"%(exclusive_list[ext_num]))
    myfiles = os.listdir('%s/%s/' % (fn, exclusive_list[ext_num]))
    files_num = len(myfiles)
    for j in range(files_num):
        fake_content.append('%s/%s/%s' % (fn, exclusive_list[ext_num], myfiles[j]))
    flen1 = len(fake_content)

    seq2 = range(0,flen1)
    rn.shuffle(seq2)
    text_file = open("val%s.txt"%(exclusive_list[ext_num]), "w")
    for i in range(10000):
        idx = targetLen - i
        text_file.write('%s 1\n'%(real_content[seq1[idx % glen1]]))
        text_file.write('%s 0\n'%(fake_content[seq2[idx % flen1]]))
    text_file.close()


