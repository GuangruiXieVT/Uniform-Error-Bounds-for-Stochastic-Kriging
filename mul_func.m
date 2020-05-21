function y = mul_func(x, n)
y = repmat(sin(9*x(1)^2)+sin((3*x(2))^2),n,1) + (2+cos(pi+sum(x)/2))*normrnd(0,1,[n,1]);
% elseif d==5
%     y = repmat(sin(9*x(1)^2)+sin(((3*x(2)+3*x(3)+3*x(4)+3*x(5))/4)^2),n,1) + (2+cos(pi+sum(x)/5))*normrnd(0,1,[n,1]);
% end
    