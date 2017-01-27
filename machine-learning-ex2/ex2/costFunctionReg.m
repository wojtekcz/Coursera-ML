function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% compute J

function [h] = h_th(theta, x)
  h = sigmoid(theta' * x);
endfunction

function [costi] = Cost_i(theta, xi, yi)
  costi = -yi * log(h_th(theta, xi)) - (1-yi) * log(1 - h_th(theta, xi));
endfunction

sum = 0;
for i=1:m,
  sum += Cost_i(theta, X(i,:)', y(i));
endfor

% with regularization
n = length(theta);
sumTheta = 0;
for j=2:n,
  sumTheta += theta(j)^2;
endfor

J = 1/m * sum + lambda/(2*m) * sumTheta;


% compute grad

function [grad] = grad_j(theta, X, y, j) 
  sum = 0;
  
  for i=1:m,
    xi = X(i,:)';
    yi = y(i);
    sum += (h_th(theta, xi) - yi) * xi(j);
  endfor

  % with regularization  
  reg = 0;
  if j > 1
    reg = lambda/m * theta(j);
  endif
  
  grad = 1/m * sum + reg;
endfunction

n = length(grad);
for j=1:n,
  grad(j) = grad_j(theta, X, y, j);
endfor

% =============================================================

end
