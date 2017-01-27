function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

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

J = 1/m * sum;


% compute grad

function [grad] = grad_j(theta, X, y, j) 
  sum = 0;
  
  for i=1:m,
    xi = X(i,:)';
    yi = y(i);
    sum += (h_th(theta, xi) - yi) * xi(j);
  endfor

  grad = 1/m * sum;
endfunction

n = length(grad);
for j=1:n,
  grad(j) = grad_j(theta, X, y, j);
endfor

% =============================================================

end
