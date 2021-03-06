clear all
clc
error_list = [];
beta_list = [];
gamma_list = [];
betamax_list = [];
betamin_list = [];
lf_list = [];
lnu_list = [];
omega_list = [];
Kinv_list = [];
term1_list = [];
term2_list = [];
smax_list = [];
smin_list = [];
lk_list = [];
sf_list = [];
ls_list = [];
hw = zeros(2500,100);
cover = zeros(2500,100);
cover1 = zeros(2500,100);
cover2 = zeros(2500,100);
cover3 = zeros(2500,100);
hw1 = zeros(2500,100);
hw2 = zeros(2500,100);
hw3 = zeros(2500,100);
s = zeros(2500,100);
t = 1;
m = 1; % index for macro-reps
while t <= 100
try
m
rng(m);
B = 2560; %total budget
K = 256; %number of design points
N = 2500; %number of prediction points
tau = 1e-4/K^2; %grid constant
E = 2; %input dimensionality
deltaL = 0.01;  
delta = 0.04;
LBSs = [-1,-1];
UBSs = [1,1];

% latin hypercube design
x = lhsdesign(K-4,2,'criterion','maximin');
d = x(:,1)*(UBSs(1)-LBSs(1))+LBSs(1); 
S = x(:,2)*(UBSs(2)-LBSs(2))+LBSs(2); 
x = [LBSs; UBSs; [LBSs(1),UBSs(2)];[UBSs(1),LBSs(2)];d,S];

xs1 = linspace(-1,1,sqrt(N))';
xs2 = linspace(-1,1,sqrt(N))';
[p,q] = meshgrid(xs1,xs2);
xs = [p(:) q(:)];


true_var = (2+cos(pi+sum(x,2)/2)).^2;
% different budget allocation schemes
lambda = true_var/sum(true_var);
%lambda = sqrt(true_var)/sum(sqrt(true_var));
NReps = ceil(lambda*B);
% NReps = ceil(B/K)*ones(K,1);  
ys = sin(9*xs(:,1).^2)+sin((3*xs(:,2)).^2);
y_raw = cell(K,1);
for i = 1:K
    y_raw{i,1} = mul_func(x(i,:), NReps(i));
end
for i = 1:K
    y_bar(i,1) = mean(y_raw{i,1});
    y_var(i,1) = var(y_raw{i,1})/NReps(i);
end
y_var = y_var + 1e-2*ones(K,1);
Ntr = length(x);
Nte = length(xs);

mf = {@meanZero};
cf = {@covSEard};
lf = {@likGauss};
sf = 1;
hyp0.mean = []; 
Ncg = 500;
ell = ones(2,1);
hyp0.cov = log([ell;sf]);
hyp = minimize(hyp0, @nlogLikelihood, -Ncg , mf , cf , x, y_bar, y_var);
ls = exp(hyp.cov(1:2));
sf = exp(hyp.cov(3));
post = infSK(hyp, mf, cf, x, y_bar, y_var);
Lk = norm(sf^2*exp(-0.5)./ls);
Lnu = Lk*sqrt(Ntr)*norm(post.alpha);
omega = sqrt(2*tau*Lk*(1+Ntr*norm(post.Kinv)*sf^2));
beta = 2*log((1+(max(max(xs))-min(min(xs)))/tau)^E/delta);
x = x';
xs = xs';
k = @(x,xp) sf^2 * exp(-0.5*sum((x-xp).^2./ls.^2,1));
dkdxi = @(x,xp,i)  -(x(i,:)-xp(i,:))./ls(i)^2 .* k(x,xp);
ddkdxidxpi = @(x,xp,i) ls(i)^(-2) * k(x,xp) +  (x(i,:)-xp(i,:))/ls(i)^2 .*dkdxi(x,xp,i);
dddkdxidxpi = @(x,xp,i) -ls(i)^(-2) * dkdxi(x,xp,i) - ls(i)^(-2) .*dkdxi(x,xp,i) ...
    +  (x(i,:)-xp(i,:))/ls(i)^2 .*ddkdxidxpi(x,xp,i);

r = max(pdist(xs')); Lfs = zeros(E,1);
for e=1:E
    maxk = max(ddkdxidxpi(xs,xs,e));
    Lkds = zeros(Nte,1);
    for nte = 1:Nte
       Lkds(nte) = max(dddkdxidxpi(xs,xs(:,nte),e));
    end
    Lkd = max(Lkds);  
    Lfs(e) = sqrt(2*log(2*E/deltaL))*maxk + 12*sqrt(6*E)*max(maxk,sqrt(r*Lkd));
end
Lfh =  norm(Lfs);
gamma = tau*(Lnu+Lfh) + sqrt(beta)*omega;
x = x';
xs = xs';
[y_pred, s2_pred] = gp_predict(hyp, mf, cf, x, y_bar, y_var, xs);

% uniform cnfidence interval
bounds_l = y_pred - sqrt(beta).*sqrt(s2_pred)-gamma;
bounds_u = y_pred + sqrt(beta).*sqrt(s2_pred)+gamma;

% uncorrected pointwise, Bonferroni corrected and Sidak corrected
% confidence interval
a = 1-(1-0.05)^(1/N);
f = norminv(1-a/2);
g = norminv(1-0.05/(2*N));
b_l1 = y_pred - 1.96*sqrt(s2_pred);
b_u1 = y_pred + 1.96*sqrt(s2_pred);
b_l2 = y_pred - f*sqrt(s2_pred);
b_u2 = y_pred + f*sqrt(s2_pred);
b_l3 = y_pred - g*sqrt(s2_pred);
b_u3 = y_pred + g*sqrt(s2_pred);
flag_l = ys >= bounds_l;
flag_u = ys <= bounds_u;
flag_l1 = ys >= b_l1;
flag_u1 = ys <= b_u1;
flag_l2 = ys >= b_l2;
flag_u2 = ys <= b_u2;
flag_l3 = ys >= b_l3;
flag_u3 = ys <= b_u3;

% delete outliers
hw(:,t) = sqrt(beta).*sqrt(s2_pred)+gamma;
if median(hw(:,t))>10
    m = m+1;
    continue
end

cover(:,t) = flag_l.*flag_u;
cover1(:,t) = flag_l1.*flag_u1;
cover2(:,t) = flag_l2.*flag_u2;
cover3(:,t) = flag_l3.*flag_u3;
s(:,t) = sqrt(s2_pred);
hw1(:,t) = 1.96*sqrt(s2_pred);
hw2(:,t) = f*sqrt(s2_pred);
hw3(:,t) = g*sqrt(s2_pred);
betamax = max(sqrt(beta).*sqrt(s2_pred));
betamin = min(sqrt(beta).*sqrt(s2_pred));
error = sqrt(mean((y_pred-ys).^2));
error_list = [error_list;error];
betamax_list = [betamax_list;betamax];
betamin_list = [betamin_list;betamin];
beta_list = [beta_list;beta];
gamma_list = [gamma_list;gamma];
lf_list = [lf_list;Lfh];
lnu_list = [lnu_list;Lnu];
omega_list = [omega_list;omega];
term1_list = [term1_list;tau*(Lnu+Lfh)];
term2_list = [term2_list;sqrt(beta)*omega];
smax_list = [smax_list;max(sqrt(s2_pred))];
smin_list = [smin_list;min(sqrt(s2_pred))];
Kinv_list = [Kinv_list;norm(post.Kinv)];
lk_list = [lk_list;Lk];
ls_list = [ls_list;ls];
sf_list = [sf_list;sf];
t = t+1;
m = m+1;
catch
m = m+1;
end
end

%calculate simultaneous coverage probability and max half width
C = mean(prod(cover));
C1 = mean(prod(cover1));
C2 = mean(prod(cover2));
C3 = mean(prod(cover3));
hw_max = max(hw);
hw1_max = max(hw1);
hw2_max = max(hw2);
hw3_max = max(hw3);
