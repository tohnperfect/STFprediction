# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 14:58:35 2014

@author: TOHN
"""

"""
assume pre-computed CNN feature
--f=(path to images)     ; images must be in the folder images/ and feature must be in the folder features/ inside (path to images)
"""

import pickle
import os
import predictSTF_tree
import matplotlib.pyplot as plt
import numpy
from skimage import color
import scipy.io as sio

import argparse

folder_path=''

parser = argparse.ArgumentParser()
parser.add_argument('--f', type=str, help='path to the images')
args = parser.parse_args()

folder_path=args.f

class_list=[#[0,0,0], #void
            #[128,0,128], #horse
            #[64,0,0], #moutain
            [128,0,0], #building 0
            [128,128,0], #tree 1
            [128,64,128], #road 2
            [0,0,128], #cow  3 
            [0,128,0], #grass 4 
            [128,128,128], #sky 5
            [0,128,128], #sheep 6 
            [192,0,0], #aeroplane 7
            [64,64,0], #body 8
            [192,128,0], #face 9
            [128,64,0], #book 10
            [64,128,0], #water  11
            [64,0,128], #car 12
            [192,0,128], #bicycle 13
            [64,128,128], #flower 14
            [192,128,128], #sign 15
            [0,64,0], #bird 16
            [0,192,0], #chair 17
            [0,192,128], #cat 18
            [128,192,128], #dog 19
            [192,64,0]] #boat 20

num_class=len(class_list)
num_tree=5

print 'load learned forests',

tree=list()
weight=list()

for t in range(num_tree):
    with open('Tree_all{0}.pik'.format(t),'rb') as f:
        [t,w]=pickle.load(f)
    tree.append(t)
    weight.append(w)

print '.done!!'

filelist=os.listdir(folder_path+'images/')
if os.path.exists(folder_path+'images/Thumbs.db'):
    filelist.remove('Thumbs.db')
    
#load categoriser
with open('CategoriserCOV.pik','rb') as f:
    Categoriser=pickle.load(f)

def newRange(val,newMin,newMax,oldMin=0,oldMax=1):
    return (newMax-newMin)/(oldMax-oldMin)*(val-oldMax)+newMax

def predictIMG(filen):    
    window_size=21    
    
    imgtest=plt.imread(folder_path+'images/'+filen)
    

    img_lab=color.rgb2lab(imgtest)#img/255.#
    ##range CIELAB = {L in [0, 100], A in [-86.185, 98.254], B in [-107.863, 94.482]}
    ###normalise it to have range [0,1]
    img_lab[:,:,0]=newRange(img_lab[:,:,0],0.,1.,0.,100.)
    img_lab[:,:,1]=newRange(img_lab[:,:,1],0.,1.,-86.185,98.254)
    img_lab[:,:,2]=newRange(img_lab[:,:,2],0.,1.,-107.863,94.482)

    
    
    height,width,channel=numpy.shape(img_lab)

    mat=sio.loadmat(folder_path+'features/'+filen[:-4]+'.mat')
    feat_vect=mat['cnn']  

    predict=Categoriser.predict_proba(feat_vect)
    ipl_list=list()
    for a in predict:
        ipl_list.append(a[0,1])
        
    ILP=numpy.array(ipl_list)
    
    ##soften the prior by alpha [1...5]
    alpha=1
    ILP=ILP**alpha
    
    predicted_imgILP=numpy.zeros(shape=numpy.shape(imgtest))    
 
    for c in range(0,width-(window_size)):
        for r in range(0,height-(window_size)):

            
            img_c=c+10
            img_r=r+10
 
            predict=numpy.zeros(shape=(len(tree),num_class))
            for t in range(len(tree)):
                val=predictSTF_tree.predict(tree[t],img_lab[r:r+window_size,c:c+window_size,:])

                predict[t,:]=val/numpy.sum(val,axis=1)
            
            #print predict            
            STF_ILP=numpy.mean(predict,axis=0)*ILP            
            predict_classILP=STF_ILP.argmax()
                           
            predicted_imgILP[img_r,img_c,0]=class_list[predict_classILP][0]/255.
            predicted_imgILP[img_r,img_c,1]=class_list[predict_classILP][1]/255.
            predicted_imgILP[img_r,img_c,2]=class_list[predict_classILP][2]/255.
            

    plt.imsave(folder_path+'results/{0}.png'.format(filen[:-4]),predicted_imgILP)


if not os.path.exists(folder_path+'results/'):
    os.mkdir(folder_path+'results/')

for filen in filelist:
    predictIMG(filen) 
    
    

