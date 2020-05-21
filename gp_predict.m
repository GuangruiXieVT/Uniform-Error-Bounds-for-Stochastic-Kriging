function [ymu, ys2] = gp_predict(hyp, mean, cov, x, y, Vhat, xs)
post = infSK(hyp, mean, cov, x, y, Vhat);
alpha = post.alpha; L = post.L; sW = post.sW;
if issparse(alpha)                  % handle things for sparse representations
nz = alpha ~= 0;                                 % determine nonzero indices
if issparse(L), L = full(L(nz,nz)); end      % convert L and sW if necessary
if issparse(sW), sW = full(sW(nz)); end
else nz = true(size(alpha,1),1); end               % non-sparse representation
if numel(L)==0                      % in case L is not provided, we compute it
K = feval(cov{:}, hyp.cov, x(nz,:));
L = chol(eye(sum(nz))+sW*sW'.*K);
end
Ltril = all(all(tril(L,-1)==0));            % is L an upper triangular matrix?
ns = size(xs,1);                                       % number of data points
nperbatch = 1000;                       % number of data points per mini batch
nact = 0;                       % number of already processed test data points
ymu = zeros(ns,1); ys2 = ymu; fmu = ymu; fs2 = ymu; lp = ymu;   % allocate mem
while nact<ns               % process minibatches of test cases to save memory
id = (nact+1):min(nact+nperbatch,ns);               % data points to process
kss = feval(cov{:}, hyp.cov, xs(id,:), 'diag');              % self-variance
Ks  = feval(cov{:}, hyp.cov, x(nz,:), xs(id,:));         % cross-covariances
ms = feval(mean{:}, hyp.mean, xs(id,:));
N = size(alpha,2);  % number of alphas (usually 1; more in case of sampling)
Fmu = repmat(ms,1,N) + Ks'*full(alpha(nz,:));        % conditional mean fs|f
fmu(id) = sum(Fmu,2)/N;                                   % predictive means
if Ltril           % L is triangular => use Cholesky parameters (alpha,sW,L)
  V  = L'\(repmat(sW,1,length(id)).*Ks);
  fs2(id) = kss - sum(V.*V,1)';                       % predictive variances
else                % L is not triangular => use alternative parametrisation
  fs2(id) = kss + sum(Ks.*(L*Ks),1)';                 % predictive variances
end
fs2(id) = max(fs2(id),0);   % remove numerical noise i.e. negative variances
Fs2 = repmat(fs2(id),1,N);     % we have multiple values in case of sampling
[Ymu, Ys2] = feval(@likGauss_sk,[],Fmu(:),Fs2(:));
ymu(id) = sum(reshape(Ymu,[],N),2)/N;          % predictive mean ys|y and ..
ys2(id) = sum(reshape(Ys2,[],N),2)/N;                          % .. variance
nact = id(end);          % set counter to index of last processed data point
end
end