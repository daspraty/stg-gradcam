import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
from sklearn.cluster import KMeans
import torch
import os
from sklearn.metrics import plot_confusion_matrix
def marker_size(resized_gcam):
    # aa=resized_gcam[i,:]
    # print(aa.shape)
    [m,n]=resized_gcam.shape
    dt=np.reshape(resized_gcam,(m*n,1))
    # print(dt.shape)
    kmeans = KMeans(n_clusters=7, random_state=0).fit(dt)
    cntrs=np.squeeze(kmeans.cluster_centers_)
    labs = kmeans.labels_
    z_ = [[0, cntrs[0]],[1, cntrs[1]], [2, cntrs[2]], [3, cntrs[3]], [4, cntrs[4]], [5, cntrs[5]], [6, cntrs[6]]]
    sorted(z_, key=lambda x: x[1])


    dt[labs == z_[0][0]] = 5**2
    dt[labs == z_[1][0]] = 10**2
    dt[labs == z_[2][0]] = 15**2
    dt[labs == z_[3][0]] = 20**2
    dt[labs == z_[4][0]] = 25**2
    dt[labs == z_[5][0]] = 30**2
    dt[labs == z_[6][0]] = 35**2
    m_size=np.reshape(dt,(m,n))
    # [m,n]=resized_gcam.shape
    # ind=np.argsort(cntrs)
    # print(cntrs)
    # print(ind)
    # print(kmeans.labels_)
    # s=np.ones((n,), dtype=int)
    # for j in range(n):
    #     if kmeans.labels_[j]==ind[0]:
    #         s[j]=20
    #     elif kmeans.labels_[j]==ind[1]:
    #         s[j]=100
    #     elif kmeans.labels_[j]==ind[2]:
    #         s[j]=400
    # dt=np.squeeze(dt)
    # return dt.astype(int)
    return m_size





def draw_skel(path,resized_gcam,tt):
    movement = np.loadtxt(path)
    bone_list = [[1, 2], [1,3], [1,4], [1,5], [1,6], [2,7], [3,8], [4,9], [5,10], [6,11], [7,12], [8,13], [9,14], [10,15], [11,16], [12,17], [13,18], [14,19], [15,20], [16,21] ]
    print(resized_gcam.shape)
    bone_list = np.array(bone_list) - 1
    number_of_postures = movement.shape[0]

    s1='/home/pratyusha/Pratyusha_workspace/project/DATA_SET/st_gcn_TrLab/skel_image/'
    s2=path.split('/')[7:11]
    s3='/'.join(s2)
    # print (s3)
    os.makedirs(s1+s3,exist_ok=True)
    sz=marker_size(resized_gcam)
    for i in range(number_of_postures):
        fig, ax = plt.subplots(1, figsize=(3, 8))
        plt.title('Skeleton')

        # skeleton = movement[i,:]

        x = movement[i, 1::3]
        y = movement[i, 2::3]
        z = movement[i, 3::3]

        mnx=np.min(x)
        mxx=np.max(x)

        mny=np.min(y)
        mxy=np.max(y)

        mnz=np.min(z)
        mxz=np.max(z)


        # plt.xlim(mnx-10, mxx+10)
        # plt.ylim(mny-10, mxy+10)

        # plt.zlim(mnz-10, mxz+10)
        # ar=marker_size(resized_gcam,i)
        # print (s)
        # s = [2*n for n in range(len(x))]
        ar=np.squeeze(sz[i,:])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        col=['r','b','g','c','m']
        sc = ax.scatter(x, y, z, s=ar)
        c=0
        for bone in bone_list:
            if c==4:
                c=0
            else:
                c=c+1
            ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], [z[bone[0]], z[bone[1]]], col[c])
        save_path=s1+s3+'/'+str(i)+'.png'
        # print(tt)
        plt.title(tt)
        plt.savefig(save_path)
        plt.clf()


def draw_skel_true_pred(path,resized_gcam_true,resized_gcam_pred,tt):
    # print(tt)
    movement = np.loadtxt(path)
    bone_list = [[1, 2], [1,3], [1,4], [1,5], [1,6], [2,7], [3,8], [4,9], [5,10], [6,11], [7,12], [8,13], [9,14], [10,15], [11,16], [12,17], [13,18], [14,19], [15,20], [16,21] ]
    # print(resized_gcam.shape)
    bone_list = np.array(bone_list) - 1
    number_of_postures = movement.shape[0]

    s1='/home/pratyusha/Pratyusha_workspace/project/DATA_SET/st_gcn_diff/skel_image/'
    s2=path.split('/')[7:11]
    s3='/'.join(s2)
    # print (s3)
    os.makedirs(s1+s3,exist_ok=True)
    sz_tr=marker_size(resized_gcam_true)
    sz_pr=marker_size(resized_gcam_pred)
    [m,n]=resized_gcam_true.shape
    sz=np.ones((m,n), dtype=int)
    sz=np.absolute((sz_pr-sz_tr))
    print(np.sum(np.sum(sz)))
    for i in range(number_of_postures):
        print(i)
        fig, ax = plt.subplots(1, figsize=(3, 8))
        # plt.title('Skeleton')

        # skeleton = movement[i,:]

        x = movement[i, 1::3]
        y = movement[i, 2::3]
        z = movement[i, 3::3]

        mnx=np.min(x)
        mxx=np.max(x)

        mny=np.min(y)
        mxy=np.max(y)

        mnz=np.min(z)
        mxz=np.max(z)


        # plt.xlim(mnx-10, mxx+10)
        # plt.ylim(mny-10, mxy+10)

        # plt.zlim(mnz-10, mxz+10)
        # ar=marker_size(resized_gcam,i)
        # print (s)
        # s = [2*n for n in range(len(x))]
        ar=np.squeeze(sz[i,:])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        col=['r','b','g','c','m']
        sc = ax.scatter(x, y, z, s=ar)
        c=0
        for bone in bone_list:
            if c==4:
                c=0
            else:
                c=c+1
            ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], [z[bone[0]], z[bone[1]]], col[c])
        # save_path=s1+s3+'/'+str(i)+'.png'
        save_path='error'+str(i)+'.png'
        # print(tt)
        plt.title(tt)
        plt.savefig(save_path)
        plt.clf()

def plot_heat_map(resized_gcam,tr_label,pr_label, sample_name, save_path):
    action_name=['drink water',
    'eat meal/snack',
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
    'put on a hat/cap',
    'take off a hat/cap',
    'cheer up',
    'hand waving',
    'kicking something',
    'reach into pocket',
    'hopping (one foot jumping)',
    'jump up',
    'make a phone call/answer phone',
    'playing with phone/tablet',
    'typing on a keyboard',
    'pointing to something with finger',
    'taking a selfie',
    'check time (from watch)',
    'rub two hands together',
    'nod head/bow',
    'shake head',
    'wipe face',
    'salute',
    'put the palms together',
    'cross hands in front (say stop)',
    'sneeze/cough',
    'staggering',
    'falling',
    'touch head (headache)',
    'touch chest (stomachache/heart pain)',
    'touch back (backache)',
    'touch neck (neckache)',
    'nausea or vomiting condition',
    'use a fan (with hand or paper)/feeling warm',
    'punching/slapping other person',
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
    # pri  nt(tr_label[0])


    l1=action_name[tr_label[0]]
    l2=action_name[pr_label[0]]
    tt='Tr Label='+l1+' ,Pr label='+l2

    os.makedirs(save_path+ l1, exist_ok = True)
    out_file = save_path+ l1 +'/'+sample_name+'_heatmap.png'

    joint_labels=['base of the spine', 'middle of the spine','neck', 'head', 'left shoulder', 'left elbow', 'left wrist', 'left hand', 'right shoulder', 'right elbow' , 'right wrist', 'right hand', 'left hip', 'left knee', 'left ankle', 'left foot', 'right hip','right knee', 'right ankle', 'right foot' ,'spine' , 'tip of the left hand', 'left thumb','tip of the right hand' ,'right thumb']
    j=[]
    for i in range(0,len(joint_labels),2):
        j.append(joint_labels[i])
    # print(len(joint_labels))

    # print(resized_gcam.shape)

    # s1='/home/pratyusha/Pratyusha_workspace/project/project_results/graph_gradcam/nturgb/heatmap/'
    # s2=path.split('/')[7:11]
    # s3='/'.join(s2)
    # # print (s3)
    # os.makedirs(s1+s3,exist_ok=True)
    # fig, ax = plt.subplots(2)

    pltx1=np.transpose(resized_gcam[0,:,:])
    sns.set()
    ax = sns.heatmap(pltx1)
    plt.xlabel('Time')
    plt.ylabel('Joints')
    # ax1.set_yticklabels(j, rotation=0,fontsize=5)

    # pltx2=np.transpose(resized_gcam[1,:,:])
    # # sns.set()
    # ax = sns.heatmap(pltx2)
    # plt.xlabel('Time')
    # plt.ylabel('Joints')
    # ax2.set_yticklabels(joint_labels, rotation=0,fontsize=5)

    # save_path=s1+s3+'.png'
    # print(tt)
    plt.title(tt+'\n Grad cam for True label')

    plt.savefig(out_file)
    plt.clf()
    plt.close('all')

def plot_rawdata(feature,save_path,tr_label):
    # feature = torch.mean(feature, 1).squeeze()
    resized_gcam=feature.data.cpu().numpy()
    # print(resized_gcam.shape)
    data=resized_gcam[0,0,:,:]**2+resized_gcam[0,1,:,:]**2+resized_gcam[0,2,:,:]**2
    action_name = np.load('action_name.npy')
    # fig, ax = plt.subplots(2)

    pltx1=np.transpose(data[:,:])
    sns.set()
    ax = sns.heatmap(pltx1)
    plt.xlabel('Time')
    plt.ylabel('Joints')
    # ax1.set_yticklabels(j, rotation=0,fontsize=5)

    # pltx2=np.transpose(resized_gcam[1,:,:])
    # # sns.set()
    # ax = sns.heatmap(pltx2)
    # plt.xlabel('Time')
    # plt.ylabel('Joints')
    # ax2.set_yticklabels(joint_labels, rotation=0,fontsize=5)
    print(tr_label)

    # print(tt)
    # plt.title(tt+'\n Grad cam for Pred label')
    plt.title('action='+action_name[tr_label[0]])
    plt.savefig(out_file)
    # plt.savefig(save_path)
    plt.clf()
    plt.close('all')


def plot_heat_map_multilayer(feature,save_path,layer):
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Daily closing prices", fontsize=18, y=0.95)

    # loop through the length of tickers and keep track of index
    for n, ticker in enumerate(tickers):
        # add a new subplot iteratively
        ax = plt.subplot(3, 2, n + 1)

        # filter df and plot ticker on the new subplot axis
        df[df["ticker"] == ticker].plot(ax=ax)

        # chart formatting
        ax.set_title(ticker.upper())
        ax.get_legend().remove()
        ax.set_xlabel("")
def confusion_mat(mat,save_path):
    action_name = np.load('action_name.npy')
    # fig, ax = plt.subplots(2)

    # plot_confusion_matrix(clf, X_test, y_test)
    sns.set()
    ax = sns.heatmap(mat)
    plt.xlabel('true')
    plt.ylabel('predicted')
    # ax.set_yticklabels(action_name, rotation=0,fontsize=5)
    # ax.set_xticklabels(action_name, rotation=90,fontsize=5)

    # pltx2=np.transpose(resized_gcam[1,:,:])
    # # sns.set()
    # ax = sns.heatmap(pltx2)
    # plt.xlabel('Time')
    # plt.ylabel('Joints')
    # ax2.set_yticklabels(joint_labels, rotation=0,fontsize=5)


    # print(tt)
    # plt.title(tt+'\n Grad cam for Pred label')
    plt.title('conf_mat')
    plt.savefig('save_path+conf_mat.png')
    # plt.savefig(save_path)
    plt.clf()
    plt.close('all')

    # import numpy as np
    # from draw_skeleton_fn import confusion_matq
    # mat=np.load('conf_mat.npy')
    # confusion_mat(mat,save_path)
