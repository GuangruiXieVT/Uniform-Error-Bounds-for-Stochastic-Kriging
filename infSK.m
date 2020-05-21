function post = infSK(hyp, mean, cov, x, y, Vhat, opt)

% Exact inference for a GP with Gaussian likelihood.
%
% Compute a parametrization of the posterior, the negative log marginal
% likelihood and its derivatives w.r.t. the hyperparameters. The function takes
% a specified covariance function (see covFunctions.m) and likelihood function
% (see likFunctions.m), and is designed to be used with gp.m.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2018-08-22.
%                                      File automatically generated using noweb.
%
% See also INFMETHODS.M, APX.M.

if nargin<7, opt = []; end                          % make sure parameter exists

[n, D] = size(x);
[m,dm] = feval(mean{:}, hyp.mean, x);           % evaluate mean vector and deriv
sn2 = 1;
W = ones(n,1)/sn2;            % noise variance of likGauss
K = apx_sk(hyp,cov,x,Vhat,opt);                        % set up covariance approximation
[ldB2,solveKiW,dW,dhyp,post.L] = K.fun(W); % obtain functionality depending on W

alpha = solveKiW(y-m);
post.alpha = K.P(alpha);                       % return the posterior parameters
post.sW = sqrt(W);                              % sqrt of noise precision vector
post.Kinv = solveKiW(eye(n));


