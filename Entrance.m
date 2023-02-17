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
lambda3=10^-2;%correlation2
lambda4=10^-2;%correlation3
rho = 10^-2; 
rRatio = 0.5;

times = 2;  % 10 times
folds = 5; % 10 fold
[num_sample, num_features] = size(features);
[~, num_labels] = size(labels);

S1 = [ones(num_sample, num_features), zeros(num_sample, num_labels)];
S2 = [zeros(num_sample, num_features), ones(num_sample, num_labels)];

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
        relationI = corrcoef([train_feature;test_feature]','Rows','complete');
        relationI(find(isnan(relationI)==1)) = 0;
        DI = sum(relationI,2);
        L0 = -1 * relationI;
        col_I = size(L0,1);
        for i=1:col_I
            L0(i,i) = DI(i,1) + relationI(i,i);
        end
             
        % feature correction
        relationF = corrcoef([train_feature;test_feature],'Rows','complete');
        relationF(find(isnan(relationF)==1)) = 0;
        DF = sum(relationF,2);
        L1 = -1 * relationF;
        col_F = size(L1,1);
        for i=1:col_F
            L1(i,i) = DF(i,1) + relationF(i,i);
        end
        L1 = [L1,zeros(num_features,num_labels); zeros(num_labels, num_features+num_labels)];

        % label correction
        relationL = corrcoef([train_distribution],'Rows','complete');
        relationL(find(isnan(relationL)==1)) = 0;
        DL = sum(relationL,2);
        L2 = -1 * relationL;
        col_L = size(L2,1);
        for i=1:col_L
            L2(i,i) = DL(i,1) + relationL(i,i);
        end
        L2 = [zeros(num_features, num_features+num_labels); zeros(num_labels, num_features), L2];

        tic
        
        % init G
        G=ones(size(Z));
        % Training
        [G,obj_value] = Train(time,fold, S0, S1,S2, Z, G,lambda1,lambda2,lambda3,lambda4, rho,rRatio, L0, L1,L2);     
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



