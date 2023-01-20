clear;
clc;
cd('./DataSets');
    load SJAFFE;
cd('../');
features = double(real(features));

% parameters setting
lambda1=10^-4;%L1
lambda2=10^-2;%correlation1
lambda3=10^-3;%correlation2
rho = 10^-2; 

times = 2;  % 10 times
fold = 10; % 10 fold
[num_sample, ~] = size(features);
for itrator=1:times
    indices = crossvalind('Kfold', num_sample, fold);
    mea=[];
    for rep=1:fold
        testIdx = find(indices == rep);
        trainIdx = setdiff(find(indices),testIdx);
        test_feature = features(testIdx,:);
        test_distribution = labels(testIdx,:);
        train_feature = features(trainIdx,:);
        train_distribution = labels(trainIdx,:);
        relation = corrcoef(train_distribution,'Rows','complete');
        D = sum(relation,2);
        L = -1 * relation;
        col = size(L,1);
        for i=1:col
            L(i,i) = D(i,1) + relation(i,i);                                               
		end
        
        relationF = corrcoef(train_feature','Rows','complete');
        relationF(find(isnan(relationF)==1)) = 0;
        D_F = sum(relationF,2);
        L_F = -1 * relationF;
        col_F = size(L_F,1);
        for i=1:col_F
            L_F(i,i) = D_F(i,1) + relationF(i,i);
        end
               
        tic
        jointW=eye(size(train_feature,2),size(train_distribution,2));
        % Training
        [weights,weight1,weight2,obj_value] = Train(itrator,rep, train_feature,train_distribution,jointW,lambda1,lambda2,lambda3,rho,L, L_F);     
        % Prediction
        pre_distribution = Predict(weights,test_feature);
        [trow,tcol]=find(isnan(pre_distribution));
        pre_distribution(trow,:)=[];
        test_distribution(trow,:)=[];
        
        cd('./measures');
        mea(rep,1)=sorensendist(test_distribution, pre_distribution);
        mea(rep,2)=kldist(test_distribution, pre_distribution);
        mea(rep,3)=chebyshev(test_distribution, pre_distribution);
        mea(rep,4)=intersection(test_distribution, pre_distribution);
        mea(rep,5)=cosine(test_distribution, pre_distribution);
        mea(rep,6)=euclideandist(test_distribution, pre_distribution);
        mea(rep,7)=squaredxdist(test_distribution, pre_distribution);
        mea(rep,8)=fidelity(test_distribution, pre_distribution);
        cd('../');
        fprintf('=========================== %d times %d cross ( %d seconds )======================= \n', itrator, rep, toc);
    end
    res_once(itrator,:) = mean(mea,1);
end
meanres=mean(res_once, 1)
stdres=std(res_once, 1)



