function [G,M,E, convergence]=Train(time,fold,S0,S1, Z, G,lambda0,lambda1,lambda2,lambda3, rho, L0, L1)
[num_ins, num_prop] = size(G); % number of instance and properties
M = G;
E = zeros(num_ins, num_prop);
gamma1 = zeros(num_ins,num_prop);
gamma2 = zeros(num_ins,1);

[row,col] = size(Z);
Ic = ones(col,1);
Ir = ones(row,1);

max_iter=30;

convergence=zeros(max_iter,1);

t=0;
% options = optimoptions(@fminunc,'Display','iter','Algorithm','trust-region','SpecifyObjectiveGradient',true);
% options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true);
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton',...
    'HessianApproximation','lbfgs', ...
    'Display','iter',...
    'MaxIterations',300, ...
    'MaxFunctionEvaluations',300, ...
    'OptimalityTolerance',1e-30, ...
    'SpecifyObjectiveGradient',true);

while(t<max_iter)
    fprintf(' \n####################### %d times %d cross %d iteretor start..===================== \n', time, fold, t);
    t=t+1;
    
    G=fminunc(@(G)ProgressG(S0,S1,  Z, G,M,E,lambda2,lambda3, rho, L0, L1, gamma1,gamma2),G,options);
    G=real(G);
    
%     U = u_solve(G, E,V,rho,gamma1);
%     
%     V = v_solve(G, E,U,rho,gamma1);
    M = m_solve(G, E, lambda0, rho,gamma1);
    
    E = fista_backtracking_lasso(lambda1, rho, G, M,E,gamma1);
    
    gamma1 = gamma1 + rho*(G-M-E);
    gamma2 = gamma2 + rho*((G .* S1)*Ic-Ir);
  
    convergence(t,1)=get_obj(t,S0,S1,  Z, G, M, E, gamma1, gamma2,lambda0,lambda1, lambda2,lambda3, rho, L0, L1,Ic, Ir);
end

filename = sprintf('./conv/%s.mat',"conv-"+time+"-"+fold);
% save filename convergence  
save (filename ,'convergence')
end


function [obj_value]=get_obj(t,S0, S1,Z, G, M, E, gamma1, gamma2,lambda0,lambda1, lambda2,lambda3, rho, L0,L1, Ic, Ir)
    % objective value
    obj_fir = norm(S0.*(Z-G), 'fro')^2;
    
    obj_sec = lambda1* sum(sum(abs(E),2),1);
    
    tp2 = G'*L0*G;
    obj_third =  lambda2*trace(tp2);
    
    obj_fourth = sum(sum(gamma1.*(G-M-E),1),2);
    obj_fifth = (rho/2)*norm(G-M-E,'fro')^2;
    
    obj_sixth = sum(sum(gamma2.*((G.*S1) * Ic - Ir),1),2);
    obj_seven = (rho/2)*norm((G.*S1) * Ic - Ir,'fro')^2;
    
    add1 = lambda3 * trace((G)*L1*(G)');

    add2 = lambda0 * sum(svd(M,'econ'));
    
    obj_value = obj_fir +obj_sec + obj_third + obj_fourth+ obj_fifth +obj_sixth+obj_seven+add1+add2;
end

% singular_value_threshold
function M1=m_solve(G, E, lambda0, rho,gamma1)
    [U,Sigma,V]=svd(E-G-gamma1./rho);
    [row,col]=size(Sigma);
    temp = zeros(size(Sigma,1),size(Sigma,2));
    for i=1:row
        for j=1:col
            temp(i,j)=max(Sigma(i,j)-lambda0/rho,0);
        end
    end
    M1 = U*temp*V';
end

