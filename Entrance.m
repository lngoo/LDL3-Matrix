clear;
clc;
cd('./DataSets');
%     load Human_Gene;
    load SJAFFE;
cd('../');
features = double(real(features));

% parameters setting
lambda1=10^-2;%L1
lambda2=10^-2;%correlation1
rho = 10^-2; 
rRatio = 1;

times = 2;  % 10 times
folds = 5; % 10 fold
[num_sample, num_features] = size(features);
[~, num_labels] = size(labels);
for time=1:times
    indices = crossvalind('Kfold', num_sample, folds);
    mea=[];
    for fold=1:folds
        testIdx = find(indices == fold);
        trainIdx = setdiff(find(indices),testIdx);
        test_feature = features(testIdx,:);
        test_distribution = labels(testIdx,:);
        train_feature = features(trainIdx,:);
        train_distribution = labels(trainIdx,:);
        
        [num_train, ~] = size(train_feature);
        [num_test, ~] = size(test_feature);
        
        % mask matrix
        S0 = ones(num_train, num_features+num_labels);
        TMP1 = ones(num_test, num_features);
        TMP2 = zeros(num_test, num_labels);
        S0 = [S0;TMP1,TMP2];

        
        % big matrix 
        Z = [train_feature, train_distribution; test_feature, test_distribution];
        Z = Z .* S0;

        % instance correction
        relationF = corrcoef([train_feature;test_feature]','Rows','complete');
        relationF(find(isnan(relationF)==1)) = 0;
        D_F = sum(relationF,2);
        L_F = -1 * relationF;
        col_F = size(L_F,1);
        for i=1:col_F
            L_F(i,i) = D_F(i,1) + relationF(i,i);
        end
               
        tic
        
        % init G
        G=ones(size(Z));
        % Training
        [G,obj_value] = Train(time,fold, S0, Z, G,lambda1,lambda2, rho,rRatio, L_F);     
        % Prediction
        pre_distribution = G(num_train+1:num_sample ,num_features+1:num_features+num_labels);
        [trow,tcol]=find(isnan(pre_distribution));
        pre_distribution(trow,:)=[];
        test_distribution(trow,:)=[];
        
        cd('./measures');
        mea(fold,1)=sorensendist(test_distribution, pre_distribution);
        mea(fold,2)=kldist(test_distribution, pre_distribution);
        mea(fold,3)=chebyshev(test_distribution, pre_distribution);
        mea(fold,4)=intersection(test_distribution, pre_distribution);
        mea(fold,5)=cosine(test_distribution, pre_distribution);
        mea(fold,6)=euclideandist(test_distribution, pre_distribution);
        mea(fold,7)=squaredxdist(test_distribution, pre_distribution);
        mea(fold,8)=fidelity(test_distribution, pre_distribution);
        cd('../');
        fprintf('=========================== %d times %d cross ( %d seconds )======================= \n', time, fold, toc);
    end
    res_once(time,:) = mean(mea,1);
end
meanres=mean(res_once, 1)
stdres=std(res_once, 1)



