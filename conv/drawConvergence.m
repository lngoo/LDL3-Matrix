clear;
clc;

load conv-1-1.mat


%% draw
plot(convergence,'LineWidth',1,...
   'Color','blue')

% ylim([0 270])
xlabel('Iterations')
ylabel('Objective Function Value')
% legend('Location','northeast')
% title('Convergence')
