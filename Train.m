function [jointW,weight1,weight2,convergence3]=Train(itrator,rep, train_feature,train_distribution,jointW,lambda1,lambda2,lambda3,rho,relationL,L_F)
[num_fea, num_class] = size(jointW);
weight1 = jointW;
weight2 = eye(num_class, num_class);
weight3 = weight1;
gamma1 = zeros(num_fea,num_class);
gamma2 = zeros(num_fea,num_class);
gamma3 = zeros(size(train_feature,1),1);

[n,m]=size(train_feature);
max_iter=100;
convergence1=zeros(max_iter,1);
convergence2=zeros(max_iter,1);
convergence3=zeros(max_iter,1);
epsilon_primal=zeros(max_iter,1);
epsilon_dual=zeros(max_iter,1);
epsilon_abs=1e-4;
epsilon_rel=1e-2;
t=0;

options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true);
while(t<max_iter)
    fprintf(' \n####################### %d times %d cross %d iteretor start..===================== \n', itrator, rep,t);
    t=t+1;
    jointW=fminunc(@(jointW)ProgressW(train_feature, train_distribution, jointW, weight1, weight2, gamma1, gamma3, lambda2, rho, relationL),jointW,options);
    jointW=real(jointW);
    
    weight1_last = weight1;
    weight1=fminunc(@(weight1)ProgressW1(train_feature, jointW, weight1, weight2, weight3, gamma1, gamma2, lambda3, rho, L_F),weight1,options);
    weight1=real(weight1);
    
    weight2_last = weight2;
    weight2 = w2_solve(jointW,weight1,gamma1,rho);
    
    weight3 = w3_solve(weight1,gamma2,lambda1,rho);
    
    gamma1 = gamma1 + rho*(jointW - weight1*weight2);
    gamma2 = gamma2 + rho*(weight1-weight3);
    [row,col] = size(train_distribution);
    Ic = ones(col,1);
    In = ones(row,1);
    gamma3 = gamma3 + rho*(train_feature*jointW*Ic-In);
  
    convergence3(t,1)=get_obj(t,train_feature, train_distribution, jointW, weight1, weight2,weight3, gamma1, gamma2,gamma3,lambda1, lambda2, lambda3, rho, relationL,L_F);
end
end


function [obj_value]=get_obj(t,train_feature, train_distribution, jointW, weight1, weight2,weight3, gamma1, gamma2,gamma3,lambda1, lambda2, lambda3, rho, relationL,L_F)
[row,col] = size(train_distribution);
Ic = ones(col,1);
In = ones(row,1);

% objective value
obj_fir = norm(train_feature * jointW - train_distribution, 'fro')^2;

L = relationL;
tp = train_feature*jointW*L*jointW'*train_feature';
obj_sec = trace(tp);

tp2 = (train_feature*weight1)'*L_F*(train_feature*weight1);
obj_third = trace(tp2);

obj_fourth = sum(sum(gamma1.*(jointW-weight1*weight2),1),2);
obj_fifth = norm(jointW - weight1*weight2,'fro')^2;

obj_sixth = sum(sum(gamma2.*(weight1 - weight3),1),2);
obj_seven = norm(weight1 - weight3,'fro')^2;

obj_eight = sum(gamma3.*(train_feature*jointW*Ic-In));
obj_nine = norm(train_feature*jointW*Ic-In,'fro')^2;

obj_L1 = sum(sum(abs(weight1),2),1);
obj_value = obj_fir + lambda1*obj_L1 + lambda2*obj_sec + lambda3*obj_third ...
       + obj_fourth+obj_sixth+obj_eight + rho*(obj_fifth + obj_seven + obj_nine)/2;
end


function [weight2] = w2_solve(jointW,weight1,gamma1,rho)
  tmp = weight1'*weight1;
  weight2 = tmp\(weight1'*jointW+1/rho*weight1'*gamma1);
end

function [weight3] = w3_solve(weight1,gamma2,lambda1,rho)
Q = weight1 + gamma2./rho;
C = lambda1/rho;
[row,col] = size(Q);
zo = zeros(row,1);
for i=1:col
    value = norm(Q(:,i));
    if value>C
        weight3(:,i) = (value - C) / value * Q(:,i);
    else
        weight3(:,i) = zo;
    end
end
end

