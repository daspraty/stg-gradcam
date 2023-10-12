import glob
import os
from tools.utils.ntu_read_skeleton import *
import cv2
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import pandas as pd
import scipy
import matplotlib.ticker as ticker



gm_files = glob.glob('/xsub/*/*/*') #action_name/filename
skel_files_path = '/NTU-RGB+D/nturgb+d_skeletons/' #skaleton data files
#input parameters
no_class=60
temporal_len_in=300
temporal_len_lastlayer=75
no_joints=25
mat=np.zeros((no_class,25))
jt_imp=np.zeros((no_class, temporal_len_lastlayer, no_joints))
dp=np.zeros((no_class))
scale_size=10000

for i in range(len(gm_files)):
    cls=int(gm_files[i].split('/')[-2])
    name=gm_files[i].split('/')[-1].split('.')[0]
    skel_files_name = skel_files_path + name + '.skeleton'
    skel = read_xyz(skel_files_name)
    c,f,j,b= skel.shape #coordinate X time frame X no joints X no body


    gm=np.load(gm_files[i])
    jt_imp[cls,:,:]+=gm[0,:,:]*scale_size
    nf = int(np.ceil((temporal_len_lastlayer*f)/temporal_len_in))
    gm_=gm[0,0:nf,:]

    j_imp = np.mean(gm_,axis=0)*scale_size
    dp[cls]+=1
    mat[cls,:]+=j_imp

for i in range(no_class):
    if dp[i]==0:
        jt_imp[i,:,:]=jt_imp[i,:,:]*dp[i]
        mat[i,:]=mat[i,:]*dp[i]
    else:
        jt_imp[i,:,:]=jt_imp[i,:,:]/dp[i]
        mat[i,:]=mat[i,:]/dp[i]
# scipy.io.savemat('/home/pratyusha/Pratyusha_workspace/project /project_results/graph_gradcam/nturgb/paper_results/xview/joint_time_imp.mat', {'joint_time_imp': jt_imp})

    mat[i,:]=(mat[i,:]- np.min(mat[i,:]))/(np.max(mat[i,:])- np.min(mat[i,:]))
mat=np.transpose(mat)
#
action_name = np.load('action_name.npy')
joint_labels=np.array(['spine-base', 'spine-mid','neck', 'head', 'L shoulder', 'L elbow', 'L wrist', 'L hand', 'R shoulder', 'R elbow' , 'R wrist', 'R hand', 'L hip', 'L knee', 'L ankle', 'L foot', 'R hip','R knee', 'R ankle', 'R foot' ,'spine' , 'L hand tip', 'L thumb','R hand tip' ,'R thumb'])
action_seq=np.array([1,2,3,4,7,10,11,12,13,14,15,18,19,20,21,22,23,25,28,29,30,31,32,33,34,35,36,37,38,39,40,41,44,45,46,47,48,49,50,52,53,54,55,56,57,58,5,6,8,9,16,17,24,26,27,42,43,51,59,60])-1
joint_seq=np.array([4,5,6,7,21,22,8,9,10,11,23,24,3,2,20,1,0,12,13,14,15,16,17,18,19])
action_name1=action_name[action_seq]
joint_labels1=joint_labels[joint_seq]
mat1=mat[:,action_seq]
mat1=mat1[joint_seq,:]



df_cm = pd.DataFrame(mat1, index = [i for i in range(1,26)],
                   columns = [i for i in range(1,61)])
# plt.set_size_inches(10, 10, forward=True)
fig.clf()
fig = plt.gcf()

fig.set_size_inches( 4,3)
# plt.figure(figsize = (7,7))
plt.tight_layout()
ax=sns.heatmap(df_cm, cbar=True)
bottom, top = ax.get_ylim()

plt.xlabel('Action name', fontsize=10)
plt.ylabel('Joints', fontsize=10)

plt.setp(ax.get_yticklabels(), rotation=90, fontsize=10)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())


out_path='output_figures/'
os.makedirs(out_path,  exist_ok = True)
out_file = out_path+'joint_imp_xsub_STGCN.pdf'


plt.savefig(out_file, dpi=100)
plt.clf()
plt.close('all')
