function x_new = fista_backtracking_lasso(lambda1, rho, G, M,E,gamma1)
% Solves the following problem via FISTA:
%  lambda1 ||E||_1 + rho/2*||G-M-E+1/rho*gamma1)||_F^2


L0 = 1.05; % initial choice of stepsize
eta = 1.01; % the constant in which the stepsize is multiplied
iter = 70; 

%% f = rho/2*||G-M-E+1/rho*gamma1)||_F^2
f = @(E) rho/2*norm(G-M-E+(1/rho)*gamma1,'fro')^2;

%% g = lambda1 ||E||_1
g = @(E) lambda1 * norm(E,1);

%% the gradient of f
grad = @(E) -1*rho*(G-M-E+(1/rho)*gamma1);

%% computer F
F = @(E) rho/2*norm(G-M-E+(1/rho)*gamma1,'fro')^2 +  lambda1 * norm(E,1);

%% shrinkage operator
S = @(tau, g) max(0, g - tau) + min(0, g + tau);

%% projection
P = @(L, y) S(lambda1/L, y - (1/L)*grad(y));

%% computer Q
Q = @(L, x, y) f(y) + (x-y)'*grad(y) + 0.5*L*norm(x-y) + g(x);

x_old = E;
y_old = E;
L_new = L0;
t_old = 1;
%% MAIN LOOP
for ii = 1:iter
%     ii
    % find i_k
    j = 1;
    while true
        L_bar = eta^j * L_new;
        if F(P(L_bar, y_old)) <= Q(L_bar, P(L_bar, y_old), y_old)
            L_new = L_new * eta^j;
            break
        else
            j = j + 1;
        end
    end
    x_new = P(L_new, y_old);
    t_new = 0.5 * (1+sqrt(1+4*t_old^2));
    del = (t_old-1)/(t_new);
    y_new = x_new + del*(x_new-x_old);
    
    x_old = x_new;
    t_old = t_new;
    y_old = y_new;
end

