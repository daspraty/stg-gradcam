
from nnk_graph import *
import glob
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from knn_basedon_dtwdist import *
import json
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings("ignore")


action_name=['drink water',
'eat meal or snack',
'brushing teeth',
'brushing hair',
'drop',
'pickup',
'throw',
'sitting down',
'standing up (from sitting position)',
'clapping',
'reading',
'writing',
'tear up paper',
'wear jacket',
'take off jacket',
'wear a shoe',
'take off a shoe',
'wear on glasses',
'take off glasses',
'put on a hat or cap',
'take off a hat or cap',
'cheer up',
'hand waving',
'kicking something',
'reach into pocket',
'hopping (one foot jumping)',
'jump up',
'make a phone call_answer phone',
'playing with phone_tablet',
'typing on a keyboard',
'pointing to something with finger',
'taking a selfie',
'check time (from watch)',
'rub two hands together',
'nod head or bow',
'shake head',
'wipe face',
'salute',
'put the palms together',
'cross hands in front (say stop)',
'sneeze or cough',
'staggering',
'falling',
'touch head (headache)',
'touch chest (stomachache or heart pain)',
'touch back (backache)',
'touch neck (neckache)',
'nausea or vomiting condition',
'use a fan (with hand or paper) or feeling warm',
'punching or slapping other person',
'kicking other person',
'pushing other person',
'pat on back of other person',
'point finger at the other person',
'hugging other person',
'giving something to other person',
'touch other persons pocket',
'handshaking',
'walking towards each other',
'walking apart from each other']

# s_act=['drink water','throw','kicking other person','walking towards each other']

# folders=glob.glob('/media/pratyusha/fe8098fd-1340-454b-bdea-df570c508220/home/pratyusha/Pratyusha_workspace/project/project_results/graph_gradcam/nturgb/alllayers/features/*/*')
folders=glob.glob('/home/pratyusha/gogo_old/home/pratyusha/Pratyusha_workspace/project/project_results/graph_gradcam/nturgb/alllayers/gradcam_values/*/*')

no_datapoint=len(folders)
k=10
num_classes=60
no_layers=10

# run on partial data
labl=[]
for i in range(no_datapoint):
    labl.append(action_name.index(folders[i].split('/')[-2]))
X_train, X_test, y_train, y_test = train_test_split(folders, labl, test_size=0.2, random_state=0)

no_datapoints=len(y_test)

err=np.zeros((no_layers,no_datapoints))
erl=np.zeros((no_layers,no_datapoints))



for l in range(no_layers):
    data_all=[]
    labels=[]
    start_time = time.time()
    for n in X_test:
        lab=n.split('/')[-2]
        labels.append(action_name.index(lab))
        # files=n+'/'+str(l)+'.npy'
        # data=np.load(files)
        files=n+'/'+str(l)+'.npz'
        data_=np.load(files)
        data=data_['arr_0'][0]
        # print(data.shape)
        s1,s2=data.shape


        data_all.append(np.reshape(data,(s2*s1)))


    encoded_label=torch.nn.functional.one_hot(torch.from_numpy(np.array(labels)), num_classes=60)
    encoded_label=encoded_label.cpu().numpy()
    data_all=np.array(data_all)


    # w,ad = get_knn_weighted_graph(data_all, k,  kernel="ip")
    wt,adj = knn_dtw(data_all, k)

    for p in range(no_datapoints):
        if np.sum(wt[p,:])!=0:
            wt[p,:]=wt[p,:]/np.max(wt[p,:])

    w= 1/wt[:,1:]
    ad=adj[:,1:]
    # print(w[0])

    for p in range(no_datapoints):
        if np.sum(w[p,:])!=0:
            w[p,:]=w[p,:]/np.sum(w[p,:])


    ll=np.array(labels)
    print(ll[ad[0]])
    ad_encoded=encoded_label[ad]  #dim no_datapoints X K X C
    ad_encoded=np.squeeze(ad_encoded)
    # w=0.1*np.ones((no_datapoints,k))
    for i in range(no_datapoints):
        # ws=np.zeros((k))
        ws=np.dot(w[i,:] , ad_encoded[i,:,:])

        loc=np.where(encoded_label[i,:]==1)
        erl[l,i]=1-ws[loc[0]]


        MSE = np.square(np.subtract(encoded_label[i,:],ws)).mean()
        err[l,i]=math.sqrt(MSE)
        # err[l,i]=np.dot(encoded_label[i,:],np.squeeze(encoded_label[i,:]-ws))
    print("--- %s seconds ---" % (time.time() - start_time))
a=np.mean(err,1)
al=np.mean(erl,1)

err_clswise=np.zeros((60,2))
err_lay_cls=np.zeros((60,no_layers))
for layer in range(no_layers):
    for ind, lab in enumerate (labels):
        err_clswise[lab,0]+=1
        err_clswise[lab,1]+=err[layer,lab]
    err_lay_cls[:,layer]=  err_clswise[:,1]/err_clswise[:,0]
np.savetxt('./results/results_aug2/error_classwise_v2.csv', err_lay_cls, delimiter=',')





# # # a=a-np.mean(a)s
# # # print(a)
# np.save('./results/errorKNN_filtweighted_feature_xsub.npy',al)
# np.save('./results/errorKNN_filtweighted_feature_xsub_al.npy',a)
# # #     # y_wy[i,:]=np.sum(encoded_label[i]-ws)
# # #
# # #
# # # # #
# a1=np.load('./results/feature_smootheness_layes_nnk10.npy')
# a2=np.load('./results/feature_smootheness_layes_nnk50.npy')
# a3=np.load('./results/feature_smootheness_layes_nnk100.npy')
# # a1=a1-np.mean(a1)
# # a2=a2-np.mean(a2)
# # a3=a3-np.mean(a3)
# plt.plot(a1, 'r', label='k=10')
# plt.plot(a2, 'b', label='k=50')
# plt.plot(a3, 'g', label='k=100')
# plt.xlabel('STGCN Layers')
# plt.ylabel('Label smoothness error')
# plt.legend(['k=10', 'k=50', 'k=100'])
# plt.savefig('./results/errorNNK_feature_xview.png')

plt.plot(a)
plt.xlabel('STGCN Layers')
plt.ylabel('Label smoothness error')
plt.savefig('./plots/errorKNN_gcam_aug1_xsub_v2.png')
plt.clf()
plt.plot(al)
plt.xlabel('STGCN Layers')
plt.ylabel('Label smoothness error')
plt.savefig('./plots/errorKNN_gcamN_aug1_xsub_al_v2.png')
