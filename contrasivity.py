import glob
import os
from tools.utils.ntu_read_skeleton import *
import cv2
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import scipy



gm_files = glob.glob('/home/pratyusha/Pratyusha_workspace/project/project_results/graph_gradcam/nturgb/npfile_prlabel_downmodel/cclassified/*/*')
skel_files_path = '/home/pratyusha/Pratyusha_workspace/project/src/office/AS-GCN/data/NTU-RGB+D/nturgb+d_skeletons/'
no_class=60
mat=np.zeros((60,25))
jt_imp=np.zeros((60,75,25))
dp=np.zeros((60))

for i in range(len(gm_files)):
    cls=int(gm_files[i].split('/')[-2])
    name=gm_files[i].split('/')[-1].split('.')[0]
    skel_files_name = skel_files_path + name + '.skeleton'
    skel = read_xyz(skel_files_name)
    c,f,j,b= skel.shape #coordinate X time frame X no joints X no body

    gm=np.load(gm_files[i])
    jt_imp[cls,:,:]+=gm[0,:,:]*10000
    nf = int(np.ceil((75*f)/300))
    gm_=gm[0,0:nf,:]

    j_imp = np.mean(gm_,axis=0)*10000
    dp[cls]+=1
    mat[cls,:]+=j_imp

for i in range(60):
    if dp[i]==0:
        jt_imp[i,:,:]=jt_imp[i,:,:]*dp[i]
        mat[i,:]=mat[i,:]*dp[i]
    else:
        jt_imp[i,:,:]=jt_imp[i,:,:]/dp[i]
        mat[i,:]=mat[i,:]/dp[i]

# scipy.io.savemat('/home/pratyusha/Pratyusha_workspace/project/project_results/graph_gradcam/nturgb/paper_results/joint_time_imp.mat', {'joint_time_imp': jt_imp})
