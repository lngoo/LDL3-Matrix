function [G,U,V,E, convergence]=Train(time,fold, S0,S1, Z, G,lambda1,lambda2,lambda3, rho)
[num_ins, num_prop] = size(G); % number of instance and properties
U = eye(num_ins, num_prop);
V = eye(num_ins, num_ins);
E = eye(num_ins, num_prop);
gamma1 = zeros(num_ins,num_prop);
gamma2 = zeros(num_ins,num_prop);
gamma3 = zeros(num_ins,1);

[row,col] = size(Z);
Ic = ones(col,1);
Ir = ones(row,1);

max_iter=50;

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
    
    G=fminunc(@(G)ProgressG(S0,S1,  Z, G,U,V,E, rho, gamma1,gamma2,gamma3),G,options);
    G=real(G);
    
    U = u_solve(lambda1,G,rho,gamma2);
    
    V = fista_backtracking_lasso(lambda2, rho, G, V,E,gamma1);
    
    E = e_solve(G, V,rho,gamma1,lambda3);
    
    gamma1 = gamma1 + rho*(G-V*G-E);
    gamma2 = gamma2 + rho*(G-U);
    gamma3 = gamma3 + rho*((G .* S1)*Ic-Ir);
  
    convergence(t,1)=get_obj(t,S0,S1, Z, G, U, V, E, gamma1, gamma2,gamma3,lambda1, lambda2,lambda3, rho,Ic, Ir);
end

filename = sprintf('./conv/%s.mat',"conv-"+time+"-"+fold);
% save filename convergence  
save (filename ,'convergence')
end


function [obj_value]=get_obj(t,S0,S1, Z, G, U, V, E, gamma1, gamma2,gamma3,lambda1, lambda2,lambda3, rho,Ic, Ir)
% objective value
obj_fir = norm(S0.*(G-Z), 'fro')^2;

obj_sec = lambda1 * sum(svd(U,'econ'));  % F-noclear

obj_third = lambda2 * sum(sum(abs(V),2),1); % F-1

obj_fourth = lambda3 * sum(sqrt(sum(E.^2,2)));  % F-21

obj_fifth = (rho/2)*norm(G-V*G-E,'fro')^2 + sum(sum(gamma1.*(G-V*G-E),1),2);
obj_sixth = (rho/2)*norm(G-U,'fro')^2 + sum(sum(gamma2.*(G-U),1),2);
obj_seven = (rho/2)*norm((G.*S1) * Ic - Ir,'fro')^2 + sum(sum(gamma3.*((G.*S1) * Ic - Ir),1),2);

obj_value = obj_fir +obj_sec + obj_third + obj_fourth+ obj_fifth +obj_sixth+obj_seven;
end

% singular_value_threshold
function M1=u_solve(lambda1,G,rho,gamma2)
    [U,Sigma,V]=svd(G-gamma2./rho);
    tao = lambda1/rho;
    [row,col]=size(Sigma);
    temp = zeros(size(Sigma,1),size(Sigma,2));
    for i=1:row
        for j=1:col
            temp(i,j)=max(Sigma(i,j)-tao,0);
        end
    end
    M1 = U*temp*V';
end

function [weight3] = e_solve(G, V,rho,gamma1,lambda3)
Q = G-V*G + gamma1./rho;
C = lambda3/rho;
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

