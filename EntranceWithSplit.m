clear;
clc;
cd('./DataSets');
    %load Emotion6u;%10^-4,10^-2,10^-3
    load SJAFFE;
%     load Twitteru;%10^-4,10^-2,10^-3
cd('../');
features = double(real(features));
% features = 0.01*features;


% parameters setting
lambda1=10^-4;%L1
lambda2=10^-2;%correlation1
lambda3=10^-3;%correlation2
rho = 10^-2;  % 0.01 ??

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    times = 1;  % 10?
    fold = 10; % 5?
    indicesName = 'indices';  % splited indexs file name
    repBegin = 1;   % which index as the test index
    resultFilename='resultsplit-sjaffe';
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
try
    load(indicesName);
catch err
    [num_sample, ~] = size(features);
    % for itrator=1:times
    indices = crossvalind('Kfold', num_sample, fold);
    save(sprintf('./%s.mat',indicesName), 'indices');
end

    load(indicesName);

    for rep=repBegin:fold

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
        [weights,weight1,weight2,obj_value] = LSTrain(1,rep, train_feature,train_distribution,jointW,lambda1,lambda2,lambda3,rho,L, L_F);
        % Prediction
        pre_distribution = LSPredict(weights,test_feature);
        [trow,tcol]=find(isnan(pre_distribution));
        pre_distribution(trow,:)=[];
        test_distribution(trow,:)=[];

        mea=[];
        meaAll = [];
        cd('./measures');
        mea(1,1)=sorensendist(test_distribution, pre_distribution);
        mea(1,2)=kldist(test_distribution, pre_distribution);
        mea(1,3)=chebyshev(test_distribution, pre_distribution);
        mea(1,4)=intersection(test_distribution, pre_distribution);
        mea(1,5)=cosine(test_distribution, pre_distribution);
        mea(1,6)=euclideandist(test_distribution, pre_distribution);
        mea(1,7)=squaredxdist(test_distribution, pre_distribution);
        mea(1,8)=fidelity(test_distribution, pre_distribution);
        cd('../');
        fprintf('=========================== %d times %d cross ( %d seconds )======================= \n', 1, rep, toc);

        try
            load(resultFilename);
        end
        [ar,ac]=size(meaAll);
		meaAll(ar+1,:)=mea;
        save (sprintf('./%s.mat',resultFilename),'meaAll');
        pause(10);  % sleep 10 s
    end
%     res_once(itrator,:) = mean(mea,1);
%     obj_value_once(itrator,1) = obj_value(size(obj_value,1),1);
% end
% meanres=mean(res_once, 1)
% stdres=std(res_once, 1)
% obj = mean(obj_value_once, 1)
% 
% save resultAll.mat meanres stdres obj res_once 

