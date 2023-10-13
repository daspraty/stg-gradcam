This is the implementation of STG-GradCAM paper to visualize the importance of Joint-time importance map for STGCNs trained on Skeleton based activity recognition dataset


Paper: Gradient-Weighted Class Activation Mapping for Spatio Temporal Graph Convolutional Network


Author: Pratyusha Das, Antonio Ortega

Link: https://ieeexplore.ieee.org/document/9746621

Please follow the steps on STGCN_README to train and test the STGCN model
Once you have the pre-procesed skeleton data and pretrained model, 

cd stg-gradcam

run python main.py recognition -c config/st_gcn/ntu-xsub/test.yaml

to generate the joint time importance map for any avtivity datapoint

Once you have the joint time importance map, you can use the code in 'plot_joint_time_importance_skeleton' folder to generate the plots
open matlab

cd plot_joint_time_importance_skeleton

run layerwise_gradcam_plot.m


