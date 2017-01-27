function [J grad] = nnCostFunction2(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end

function g = sigmoidGradient(z)
    g = zeros(size(z));

    a_l = sigmoid(z);
    g = a_l .* (1 - a_l);
end

% add 1's column for a10 bias units
X = [ones(m, 1) X];

Theta2T = (Theta2)';

% compute gradients

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

sum_J = 0;
for t = 1:m,

    x = X(t,:)';

    % compute gradients
    % 1. feedforward pass
    a1 = x;
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    %z3 = Theta2 * a2;
    a3 = sigmoid(Theta2 * a2);%z3);

    % compute J(Theta)
    yVec = zeros(num_labels, 1); yVec(y(t)) = 1;
    %hThVec = a3;
    cost_i = -yVec .* log(a3) - (1-yVec) .* log(1 - a3);
    sum_J += sum(cost_i);

    % 2. compute d3
    d3 = a3 - yVec;

    % 3. compute d2
    d2 = Theta2T * d3 .* sigmoidGradient([1;z2]);

    % 4. accumulate the gradient
    d2 = d2(2:end);
    D2 += d3*(a2)';
    D1 += d2*(a1)';
endfor

% J
J = 1/m * sum_J;

% with regularization
sumTheta1 = sum(sum(Theta1(:,2:end).^2));
sumTheta2 = sum(sum(Theta2(:,2:end).^2));
J += lambda/(2*m) * (sumTheta1 + sumTheta2);

% gradients
% 5. obtain gradient
Theta2_grad = (1/m) .* D2;
Theta1_grad = (1/m) .* D1;

% regularize gradients
Theta2_grad(:,2:end) += (lambda/m) * Theta2(:,2:end);
Theta1_grad(:,2:end) += (lambda/m) * Theta1(:,2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
