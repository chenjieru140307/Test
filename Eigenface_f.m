function [disc_set,disc_value,Mean_Image]=Eigenface_f(Train_SET,Eigen_NUM)
% 
% Train_SET:  each column vector is a sample
% the magnitude of eigenvalues of this function is corrected right !!!!!!!!!
% Centralized PCA
[NN,Train_NUM]=size(Train_SET); %样本维数*样本个数
if NN<=Train_NUM % for big sample size case   样本数目比样本长度大时    
   Mean_Image=mean(Train_SET,2);  
   Train_SET=Train_SET-Mean_Image*ones(1,Train_NUM);
   R=Train_SET*Train_SET'/(Train_NUM-1);  
   [V,S]=Find_K_Max_Eigen(R,Eigen_NUM);
   disc_value=S;
   disc_set=V;
else % for small sample size case    %一般属于此种情况，样本数量远小于样本维数
   Mean_Image=mean(Train_SET,2);  %计算样本的每一维的均值
   Train_SET=Train_SET-Mean_Image*ones(1,Train_NUM);%去均值
  R=Train_SET'*Train_SET/(Train_NUM-1);  %协方差矩阵
  [V,S]=Find_K_Max_Eigen(R,Eigen_NUM);%对协方差矩阵特征值分解，取最大的几个特征值，这里是100个
  disc_value=S;
  disc_set=zeros(NN,Eigen_NUM);  
  Train_SET=Train_SET/sqrt(Train_NUM-1);
  for k=1:Eigen_NUM
    disc_set(:,k)=(1/sqrt(disc_value(k)))*Train_SET*V(:,k);
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Eigen_Vector,Eigen_Value]=Find_K_Max_Eigen(Matrix,Eigen_NUM)
[NN,NN]=size(Matrix);
[V,S]=eig(Matrix); %Note this is equivalent to; [V,S]=eig(St,SL); also equivalent to [V,S]=eig(Sn,St); % %特征值分解
S=diag(S);%成为列向量
[S,index]=sort(S); %按照特征值大小排序（由小到大）
Eigen_Vector=zeros(NN,Eigen_NUM);%原始维数*新维数
Eigen_Value=zeros(1,Eigen_NUM);
p=NN;
for t=1:Eigen_NUM
    Eigen_Vector(:,t)=V(:,index(p)); %只取到经过排序的最大特征值对应向量
    Eigen_Value(t)=S(p);
    p=p-1;%（因为是由小到大排序，所以要递减p）
end
