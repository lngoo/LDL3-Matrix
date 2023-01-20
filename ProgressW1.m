function [obj_value,obj_grad]=ProgressW1(train_feature, jointW, weight1, weight2, weight3, gamma1, gamma2, lambda3, rho, relationLF)
    L = relationLF; % n * n

    % objective value
    tp = (train_feature*weight1)'*L*train_feature*weight1;   
    obj_fir = trace(tp);
    
    obj_sec = sum(sum(gamma1.*(jointW-weight1*weight2),1),2);
    obj_third = norm(jointW - weight1*weight2,'fro')^2;
    
    obj_fourth = sum(sum(gamma2.*(weight1-weight3),1),2);
    obj_fifth = norm(weight1-weight3,'fro')^2;
    
    obj_value = lambda3*obj_fir + obj_sec + obj_fourth + rho*(obj_third + obj_fifth)/2;
    
    % objective grad
    grad_fir = lambda3 * (train_feature' * (L'+L) * train_feature) * weight1;
    grad_sec = -1*gamma1 * weight2;
    grad_third = -1*rho*(jointW - weight1* weight2)*weight2';
    grad_fourth = gamma2;
    grad_fifth = rho*(weight1-weight3);
    obj_grad = grad_fir + grad_sec + grad_third + grad_fourth + grad_fifth;
end
