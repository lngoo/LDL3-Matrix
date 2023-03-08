function [obj_value,obj_grad]=ProgressG(S0, S1, Z, G,M,E,lambda2,lambda3, rho, L0,L1, gamma1, gamma2)
    [row,col] = size(Z);
    Ic = ones(col,1);
    Ir = ones(row,1);

    % objective value
    temp = S0 .* (Z-G);
    temp = real(temp);
    temp(isnan(temp)) = 0;
    obj_fir = norm(temp, 'fro')^2;
    
    tp = G'*L0*G;   
    obj_sec = trace(tp);

    tp2 = (G) * L1 *(G)';
    obj_third = trace(tp2);

    obj_four = norm(G-M-E+(1/rho)*gamma1,'fro')^2;
    obj_five = norm((G .* S1)*Ic-Ir+(1/rho)*gamma2,'fro')^2;
    
    obj_value = obj_fir + lambda2*obj_sec + lambda3*obj_third  + rho* (obj_four+obj_five)/2;
    
    % objective grad
    grad_fir = -2*(S0.*(Z-G));
    grad_sec = lambda2 * ((L0'+L0)*G);
    grad_third = lambda3 * (G*(L1'+L1));
    grad_four = rho*(G-M-E+(1/rho)*gamma1);
    grad_five = rho*((((G.*S1)*Ic-Ir+(1/rho)*gamma2)*Ic').*S1);
    
    obj_grad = grad_fir + grad_sec + grad_third + grad_four + grad_five;
end
