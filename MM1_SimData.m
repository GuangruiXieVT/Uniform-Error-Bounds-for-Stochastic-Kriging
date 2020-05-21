function Y_temp = MM1_SimData(Xdesign, NReps)
%  Monte Carlo simulation to generate output data at a given design point that imitate those
%  generated from an M/M/1 queue
%  
%------------------------------------------------------------------------
%  Created on 1/2/2015, last update on 1/29/2017
%------------------------------------------------------------------------
 
 
%true function f(x) = x./(1-x), where Var[epsilon(x)]= 2*x.*(1+x)/T./(1-x).^4;

% Note: R = normrnd(mu,sigma) generates random numbers from the normal distribution with mean parameter mu and standard deviation parameter sigma.
 Y_temp = zeros(NReps,1);

 T = 10^3;
 Y_temp = Xdesign./(1-Xdesign)+normrnd(0, sqrt(2*Xdesign.*(1+Xdesign)/T./(1-Xdesign).^4),[NReps,1]);
  
