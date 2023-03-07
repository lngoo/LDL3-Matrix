function [G, convergence]=Train(time,fold,S0, Z, G,lambda1,lambda2,lambda3, rho, rRatio, L0, L1)
[num_ins, num_prop] = size(G); % number of instance and properties
r = ceil(num_prop * rRatio);
U = eye(num_ins, r);
V = eye(r, num_prop);
E = eye(num_ins, num_prop);
gamma1 = zeros(num_ins,num_prop);
gamma2 = zeros(num_ins,1);

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
    
    G=fminunc(@(G)ProgressG(S0,  Z, G,lambda2,lambda3, rho, L0, L1, gamma2),G,options);
    G=real(G);
    
    U = u_solve(G, E,V,rho,gamma1);
    
    V = v_solve(G, E,U,rho,gamma1);
    
%     E = fista_backtracking_lasso(lambda1, rho, G, U,V,E,gamma1);
    
    gamma1 = gamma1 + rho*(G-U*V-E);
    gamma2 = gamma2 + rho*(G*Ic-Ir);
  
    convergence(t,1)=get_obj(t,S0,  Z, G, U, V, E, gamma1, gamma2,lambda1, lambda2,lambda3, rho, L0, L1,Ic, Ir);
end
end


function [obj_value]=get_obj(t,S0, Z, G, U, V, E, gamma1, gamma2,lambda1, lambda2,lambda3, rho, L0,L1, Ic, Ir)
% objective value
obj_fir = norm(S0.*(Z-G), 'fro')^2;

% L = relationL;
% tp = train_feature*jointW*L*jointW'*train_feature';
% obj_sec = trace(tp);

obj_sec = lambda1* sum(sum(abs(E),2),1);

tp2 = G'*L0*G;
obj_third =  lambda2*trace(tp2);

obj_fourth = sum(sum(gamma1.*(G-U*V-E),1),2);
obj_fifth = (rho/2)*norm(G-U*V-E,'fro')^2;

obj_sixth = sum(sum(gamma2.*(G * Ic - Ir),1),2);
obj_seven = (rho/2)*norm(G * Ic - Ir,'fro')^2;

add1 = lambda3 * trace((G)*L1*(G)');
% add2 = lambda4 * trace((S2.*G)*L2*(S2.*G)');

obj_value = obj_fir +obj_sec + obj_third + obj_fourth+ obj_fifth +obj_sixth+obj_seven+add1;
end

function [U] = u_solve(G, E,V,rho,gamma1)
  U = (G-E+(1/rho)*gamma1)*V'/(V*V');
end

function [V] = v_solve(G, E,U,rho,gamma1)
  V = (U'*U)\U'*(G-E+(1/rho)*gamma1);
end

% function [weight3] = w3_solve(weight1,gamma2,lambda1,rho)
% Q = weight1 + gamma2./rho;
% C = lambda1/rho;
% [row,col] = size(Q);
% zo = zeros(row,1);
% for i=1:col
%     value = norm(Q(:,i));
%     if value>C
%         weight3(:,i) = (value - C) / value * Q(:,i);
%     else
%         weight3(:,i) = zo;
%     end
% end
% end

