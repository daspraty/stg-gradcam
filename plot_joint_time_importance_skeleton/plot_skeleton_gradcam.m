function plot_skeleton_gradcam(path,l)
% clc;
% clear all; close all;
skel = read_skeleton_file('kickingotherperson_A51.skeleton');
action=9;
joint_imp=load('joint_imp.mat');
a=joint_imp.joint_imp;
joint_time_imp=joint_imp_func(path);
b1=corrcoef(a');

[nt,nj]=size(joint_time_imp);
k=0;p=0;
for i=1:60
    for j = i+1:60
        k=k+1;
        p=p+b1(i,j);
    end
end

temporal_len_input=300;


for i=1:60    
    for k=1:25
        if a(i,k)<0.85
            b(i,k)=1;
        else
            b(i,k)=0;
        end
    end
end
                
 


for i = 1:60
    for j = 1:60
        dis(i,j)=sqrt(mean(mean((a(i,:,:)-a(j,:,:)).^2)));
    end
end



SkeletonConnectionMap = [ [4 3];  % Neck
                          [3 21]; % Head
                          [21 2]; % Right Leg
                          [2 1];
                          [21 9];
                          [9 10];  % Hip
                          [10 11];
                          [11 12]; % Left Leg
                          [12 24];
                          [12 25];
                          [21 5];  % Spine
                          [5 6];
                          [6 7];   % Left Hand
                          [7 8];
                          [8 22];
                          [8 23];
                          [1 17];
                          [17 18];
                          [18 19];  % Right Hand
                          [19 20];
                          [1 13];
                          [13 14];
                          [14 15];
                          [15 16];
                        ];
colors = ['r';'g';'b';'c';'y';'m'];

for i=1:length(skel)
    for j=1:25
        data(j,i,1)=skel(i).bodies(1).joints(j).x;
        data(j,i,2)=skel(i).bodies(1).joints(j).y;
        data(j,i,3)=skel(i).bodies(1).joints(j).z;
    end
end
f=length(skel);

nBodies=1;
nf = ceil((nt*f)/temporal_len_input);


joint_time_imp=(joint_time_imp-min(joint_time_imp(:)))./(max(joint_time_imp(:))-min(joint_time_imp(:)));
jimp(:,:)=joint_time_imp(1:nf,:).*5;

nf_=round(nf/3);

jimp1=mean(jimp(1:nf_,:),1);
jimp2=mean(jimp(nf_+1:2*nf_,:),1);
jimp3=mean(jimp(2*nf_+1:end,:),1);



JointIndices1(:,:)=data(:,1,:);
JointIndices2(:,:)=data(:,round(f/2)+5,:);
JointIndices3(:,:)=data(:,end,:);




jimp1(jimp1==0)=0.1;
jimp2(jimp2==0)=0.1;
jimp3(jimp3==0)=0.1;

h=figure;
subplot(1,3,1)
plotSkel_wJointImp(JointIndices1,SkeletonConnectionMap,jimp1)
title('Start action', 'Fontsize', 20)
subplot(1,3,2)
plotSkel_wJointImp(JointIndices2,SkeletonConnectionMap,jimp2)
title('Mid action', 'Fontsize', 20)
subplot(1,3,3)
plotSkel_wJointImp(JointIndices3,SkeletonConnectionMap,jimp3)
title('End action', 'Fontsize', 20)
op_path='/plots/';
sgtitle(strcat('Action = kicking other person, layer = ',num2str(l)), 'Fontsize', 30) 
sp=strcat(op_path,'kicking_other_person',num2str(l-1),'.png');
% saveas(h, sp)
% close all


function plotSkel_wJointImp(colorJointIndices,SkeletonConnectionMap,jimp)

for i = 1:24
     for body = 1:1
%          ms=ceil(jimp(SkeletonConnectionMap(i,1))*10);
         X1 = [colorJointIndices(SkeletonConnectionMap(i,1),1,body) colorJointIndices(SkeletonConnectionMap(i,2),1,body)];
         Y1 = [colorJointIndices(SkeletonConnectionMap(i,1),2,body) colorJointIndices(SkeletonConnectionMap(i,2),2,body)];
         line(X1,Y1, 'LineWidth',3, 'LineStyle', '-', 'Marker', 'o','MarkerSize', 1, 'Color', 'r');         
     end
    hold on;
end

for i = 1 : 25
    ms=ceil(jimp(i)*10);
    plot(colorJointIndices(i,1),colorJointIndices(i,2),'Marker', 'o','MarkerSize', ms,'MarkerFaceColor', 'b');   
end
hold off;
axis off;

end

function jimp=joint_imp_func(path)
im = readNPY(path);
jimp(:,:)=im(2,:,:);
[m,n]=size(jimp);

jimp=(jimp-min(jimp(:)))./(max(jimp(:))-min(jimp(:)));
end

end

