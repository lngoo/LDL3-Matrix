function [obj_value,obj_grad]=ProgressG(S0,S1,S2,  Z, G,U,V, rho, gamma1,gamma2,gamma3)
    [row,col] = size(Z);
    Ic = ones(col,1);
    Ir = ones(row,1);

    % objective value
    temp = S0 .* (G-Z);
    temp = real(temp);
    temp(isnan(temp)) = 0;
    obj_fir = norm(temp, 'fro')^2;
    
    obj_sec = norm(G-(V.*S2)*G+(1/rho)*gamma1,'fro')^2;
    
    obj_third = norm(G-U+(1/rho)*gamma2,'fro')^2;

    obj_four = norm((G .* S1)*Ic-Ir+(1/rho)*gamma3,'fro')^2;
    
    obj_value = obj_fir  + rho* (obj_sec+obj_third+obj_four)/2;
    
    % objective grad
    grad_fir = 2*(S0.*(G-Z));
    I = eye(row);
    grad_sec = rho * ((I-(V.*S2))'*(G-(V.*S2)*G+(1/rho)*gamma1));
    grad_third = rho * (G-U+(1/rho)*gamma2);
    grad_four = 2 *rho*((((G.*S1)*Ic-Ir+(1/rho)*gamma3)*Ic').*S1);
    
    obj_grad = grad_fir + grad_sec + grad_third + grad_four;
end
