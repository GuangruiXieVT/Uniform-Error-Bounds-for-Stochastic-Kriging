clear all
clc
for m = 1:100
m
rng(m);
B = 25600; %total budget
K = 32; %number of design points
N = 1000; %number of prediction points
alpha = 0.05; 
x = linspace(0.3,0.9,K)'; %design points
true_var = 2*x.*(1+x)/1000./(1-x).^4; %true variance at each design point
% %unequal allocation
% lambda = sqrt(true_var)/sum(sqrt(true_var));
% NReps = ceil(lambda*B);

%equal alloccation
NReps = ceil(B/K)*ones(K,1); %number of reps
xs = linspace(0.3,0.9,N)'; %prediction points
ys = xs./(1-xs); %true function values at prediction points
y_raw = cell(K,1); 
for i = 1:K
    y_raw{i,1} = MM1_SimData(x(i), NReps(i)); %raw data
end
for i = 1:K
    y_bar(i,1) = mean(y_raw{i,1}); %sample mean
    y_var(i,1) = var(y_raw{i,1})/NReps(i); % sample variance
end
y_var = y_var + 1e-3*ones(K,1); %adding nugget to avoid numerical issues

%use GPML for fitting GP
mf = {@meanZero};
cf = {@covSEard};
lf = {@likGauss};
inf = {@infGaussLik_ww};
sf = 1;
hyp0.mean = []; 
Ncg = 500;
ell = ones(1,1);
hyp0.cov = log([ell;sf]);
hyp = minimize(hyp0, @nlogLikelihood, -Ncg , mf , cf , x, y_bar, y_var);
ls = exp(hyp.cov(1));
sf = exp(hyp.cov(2));
[y_pred, s2_pred] = gp_predict(hyp, mf, cf, x, y_bar, y_var, xs);
[y_x, ~] = gp_predict(hyp, mf, cf, x, y_bar, y_var, x);

% calculate the new type of uniform bounds
k = @(x,xp) exp(-0.5*(x-xp).^2./ls.^2);
kernel_mat = k(repmat(x,1,K),repmat(x',K,1));
a = sqrt(det(1/sf^2*diag(y_var)+kernel_mat));
b = alpha*sqrt(det(1/sf^2*diag(y_var)));
fk = sqrt(y_x'*kernel_mat*y_x);
beta = sqrt(2*log(a/b))+fk/sf;
bounds_l = y_pred - beta*sqrt(s2_pred);
bounds_u = y_pred + beta*sqrt(s2_pred);


flag_l = ys >= bounds_l;
flag_u = ys <= bounds_u;
cover(:,m) = flag_l.*flag_u;
hw(:,m) = beta*sqrt(s2_pred);

% figure
% % scatter(x,y_bar,'filled','k')
% hold on
% plot(xs,ys,'b','LineWidth',1.5)
% plot(xs,y_pred,'r','LineWidth',1.5)
% plot(xs,bounds_l,'--r','LineWidth',1.5)
% plot(xs,bounds_u,'--r','LineWidth',1.5)
% grid on
% legend('True function','Point prediction','Confidence interval')
% xlabel('X')
% ylabel('Y')
% title('B=25600,K=32')
% axis([0.3 0.9 0 10])
end
C = mean(prod(cover)); % calculate coverage probability
maxh = max(hw); % maximum half width

