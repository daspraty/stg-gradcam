clc;
clear all;
close all;
source_path='/grad_Cam_STGCN/skeleton/layerwise_kicking_other_person/S012C002P007R002A051/';
folder_dir=dir(source_path);
nbfile=length(folder_dir)-2;
for n=2 :nbfile
    file_name=folder_dir(n+2).name;
    file_path_complete = strcat(source_path,file_name);
    plot_skeleton_gradcam(file_path_complete,n-1);
end