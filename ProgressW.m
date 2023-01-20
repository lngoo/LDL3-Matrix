function [obj_value,obj_grad]=ProgressW(train_feature, train_distribution, jointW, weight1, weight2, gamma1, gamma3, lambda2, rho, relationL)
    [row,col] = size(train_distribution);
    Ic = ones(col,1);
    In = ones(row,1);
    L = relationL;

    % objective value
    temp = train_feature * jointW - train_distribution;
    temp = real(temp);
    temp(isnan(temp)) = 0;
    obj_fir = norm(temp, 'fro')^2;
    
    tp = train_feature*jointW*L*jointW'*train_feature';   
    obj_sec = trace(tp);
    
    obj_third = sum(sum(gamma1.*(jointW-weight1*weight2),1),2);
    obj_fourth = norm(jointW - weight1*weight2,'fro')^2;
    
    obj_fifth = sum(gamma3.*(train_feature*jointW*Ic-In));
    obj_sixth = norm(train_feature*jointW*Ic-In,'fro')^2;

    obj_value = obj_fir + lambda2*obj_sec + obj_third + obj_fifth + rho*(obj_fourth + obj_sixth)/2;
    
    % objective grad
    temp = train_feature * jointW - train_distribution;
    temp = real(temp);
    temp(isnan(temp)) = 0;
    grad_fir = 2*(train_feature'*temp);
    grad_sec = lambda2 * (train_feature' * train_feature * jointW * (L'+L));
    
    grad_third = gamma1;
    grad_fourth = rho*(jointW - weight1* weight2);
    
    grad_fifth = train_feature'*gamma3*Ic';
    grad_sixth = train_feature'*rho*(train_feature*jointW*Ic-In)*Ic';
    obj_grad = grad_fir + grad_sec + grad_third + grad_fourth + grad_fifth + grad_sixth;
end
