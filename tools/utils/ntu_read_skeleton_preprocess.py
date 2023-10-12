import numpy as np
import os


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    
    y=data

    y[:,:,0,:]=np.sqrt(2)*data[:,:,0,:]
    y[:,:,1,:]=np.sqrt(2)*data[:,:,1,:]
    y[:,:,2,:]=np.sqrt(2)*data[:,:,2,:]
    y[:,:,3,:]=np.sqrt(2)*data[:,:,3,:]
    y[:,:,20,:]=np.sqrt(2)*data[:,:,20,:]

    y[:,:,4,:]=data[:,:,4,:]+data[:,:,8,:]
    y[:,:,5,:]=data[:,:,5,:]+data[:,:,9,:]
    y[:,:,6,:]=data[:,:,6,:]+data[:,:,10,:]
    y[:,:,7,:]=data[:,:,7,:]+data[:,:,11,:]
    y[:,:,21,:]=data[:,:,21,:]+data[:,:,23,:]
    y[:,:,22,:]=data[:,:,22,:]+data[:,:,24,:]
    y[:,:,12,:]=data[:,:,12,:]+data[:,:,16,:]
    y[:,:,13,:]=data[:,:,13,:]+data[:,:,17,:]
    y[:,:,14,:]=data[:,:,14,:]+data[:,:,18,:]
    y[:,:,15,:]=data[:,:,15,:]+data[:,:,19,:]


    y[:,:,8,:]=data[:,:,4,:]-data[:,:,8,:]
    y[:,:,9,:]=data[:,:,5,:]-data[:,:,9,:]
    y[:,:,10,:]=data[:,:,6,:]-data[:,:,10,:]
    y[:,:,11,:]=data[:,:,7,:]-data[:,:,11,:]
    y[:,:,23,:]=data[:,:,21,:]-data[:,:,23,:]
    y[:,:,24,:]=data[:,:,22,:]-data[:,:,24,:]

    y[:,:,16,:]=data[:,:,12,:]-data[:,:,16,:]
    y[:,:,17,:]=data[:,:,13,:]-data[:,:,17,:]
    y[:,:,18,:]=data[:,:,14,:]-data[:,:,18,:]
    y[:,:,19,:]=data[:,:,15,:]-data[:,:,19,:]


    return y
