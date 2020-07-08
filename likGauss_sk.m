function [ymu,ys2] = likGauss_sk(y, mu, s2, inf, i)

% likGauss - Gaussian likelihood function for regression. The expression for the 
% likelihood is 
%   likGauss(t) = exp(-(t-y)^2/2*sn^2) / sqrt(2*pi*sn^2),
% where y is the mean and sn is the standard deviation.
%
% The hyperparameters are:
%
% hyp = [  log(sn)  ]
%
% Several modes are provided, for computing likelihoods, derivatives and moments
% respectively, see likFunctions.m for the details. In general, care is taken
% to avoid numerical issues when the arguments are extreme.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2018-08-22.
%                                      File automatically generated using noweb.
%
% See also LIKFUNCTIONS.M.


sn2 = 0;
if isempty(y),  y = zeros(size(mu)); end
ymu = {}; ys2 = {};
ymu = mu;                                                   % first y moment
ys2 = s2 + sn2;                                          % second y moment



