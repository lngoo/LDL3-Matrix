clear;
clc;
cd('./DataSets');
%     load Human_Gene;
     load SJAFFE;
%     load Yeast_cold;
%     load SBU_3DFE;
cd('../');
features = double(real(features));

% parameters setting
lambda0=10^-3;%L1
lambda1=10^-3;%L1
lambda2=10^-3;%instance correlation1
lambda3=10^-3;%feature correlation2 and label correlation
rho = 10^-1; 

times = 1;  % 10 times
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
        relationI = corrcoef([train_feature;test_feature]','Rows','complete');
        relationI(find(isnan(relationI)==1)) = 0;
        DI = sum(relationI,2);
        L0 = -1 * relationI;
        col_I = size(L0,1);
        for i=1:col_I
            L0(i,i) = DI(i,1) + relationI(i,i);
        end
             
        % feature correction and label correction
        relationF = corrcoef([train_feature;test_feature],'Rows','complete');
        relationF(find(isnan(relationF)==1)) = 0;

        relationL = corrcoef([train_distribution],'Rows','complete');
        relationL(find(isnan(relationL)==1)) = 0;

%         relationFL = [relationF, zeros(num_features, num_labels); zeros(num_labels, num_features), relationL];
        relationFL = [zeros(size(relationF)), zeros(num_features, num_labels); zeros(num_labels, num_features), relationL];

        DF = sum(relationFL,2);
        L1 = -1 * relationFL;
        col_F = size(L1,1);
        for i=1:col_F
            L1(i,i) = DF(i,1) + relationFL(i,i);
        end

        tic
        
        G=ones(size(Z));
        S1 = [zeros(num_sample, num_features), ones(num_sample, num_labels)];
        % Training
        [G,M,E,obj_value] = Train(time,fold, S0,S1,  Z, G,lambda0,lambda1,lambda2,lambda3, rho, L0, L1);     
        % Prediction
        K = (M+E);
        pre_distribution = K(num_train+1:num_sample ,num_features+1:num_features+num_labels);
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
        mea
        cd('../');
        fprintf('=========================== %d times %d cross ( %d seconds )======================= \n', time, fold, toc);
    end
    res_once(time,:) = mean(mea,1);
end
fprintf('sorensen, kl, chebyshev, intersection, cosine, euclidean, squaredx, fidelity \n');
meanres=mean(res_once, 1)
stdres=std(res_once, 1)



