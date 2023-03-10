function [obj_value,obj_grad]=ProgressG(S0, S1,S2, Z, G,lambda2,lambda3, rho, L0,L1, gamma2)
    [row,col] = size(Z);
    Ic = ones(col,1);
    Ir = ones(row,1);

    % objective value
    temp = S0 .* (Z-G);
    temp = real(temp);
    temp(isnan(temp)) = 0;
    obj_fir = norm(temp, 'fro')^2;
    
    tp = (G.*S1)'*L0*(G.*S1);   
    obj_sec = trace(tp);

    tp2 = (G.*S2) * L1 *(G.*S2)';
    obj_third = trace(tp2);

    
    obj_four = norm((G .* S1)*Ic-Ir+(1/rho)*gamma2,'fro')^2;
    
    obj_value = obj_fir + lambda2*obj_sec + lambda3*obj_third  + rho* obj_four/2;
    
    % objective grad
    grad_fir = -2*(S0.*(Z-G));
    grad_sec = lambda2 * ((L0'+L0)*(G.*S1).*S1);
    grad_third = lambda3 * ((G.*S2)*(L1'+L1).*S2);
    grad_four = 2 *rho*((((G.*S1)*Ic-Ir+(1/rho)*gamma2)*Ic').*S1);
    
    obj_grad = grad_fir + grad_sec + grad_third + grad_four;
end
