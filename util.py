#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def generateLoop(f_list, nb_loop=1):
    img = f_list[0]
    h, w = img.shape[0], img.shape[1]
    N = len(f_list)
    nb_fadeframe = int(N/2)

    t = 0.
    img_list = np.zeros((N-nb_fadeframe+1,h,w,3))
    wsum_list = np.zeros(N-nb_fadeframe+1)
    for i in range(N):
        img = f_list[i]
        img = img.astype(np.float64)
        if i<nb_fadeframe:
            t+=1./nb_fadeframe
        elif i>N-nb_fadeframe:
            t-=1./nb_fadeframe
        w = 1.-(1.-t**1.)**1.
        if i>N-nb_fadeframe:
            img_list[i-N+nb_fadeframe-1] += w*img
            wsum_list[i-N+nb_fadeframe-1] += w
        else:
            img_list[i] += w*img
            wsum_list[i] += w
    
    V_mloop = list()
    for loop in range(nb_loop):
        for i, img in enumerate(img_list):
            final_img = img/wsum_list[i]
            final_img = final_img.astype(np.uint8)
            V_mloop.append(final_img)
    return V_mloop
            
def videoWrite(f_list, out_path = "./output.avi", fps = 30.):
    
    img = f_list[0]
    h, w = img.shape[0], img.shape[1]
    cap = cv2.VideoCapture(0)
    fourcc = cv2.cv.CV_FOURCC(*'XVID') if cv2.__version__[0] == '2' else cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path ,fourcc, fps, (w,h))
    for img in f_list:
        out.write(img)
    
    cap.release()
    out.release()

def normalize(input):
    return input.astype(np.float32)/127.5-1.

def denormalize(input):
    return ((input+1.)*127.5).astype(np.uint8)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def sort_motion_codebook(codebook, n_clusters=100):
    X = codebook[:]
    n_clusters = min(X.shape[0], n_clusters)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    X = kmeans.cluster_centers_
    pca = PCA(n_components=1)
    pca.fit(X)
    X_pc = pca.transform(X) 
    X = X[X_pc[:,0].argsort()]
    return X

def sort_appearance_codebook(codebook):
    X = codebook[:]
    X_mean = np.array([code.mean(axis=0) for code in X])
    pca = PCA(n_components=1)
    pca.fit(X_mean)
    X_mean_pc = pca.transform(X_mean)
    X = [X[i] for i in X_mean_pc[:,0].argsort()]
    return X