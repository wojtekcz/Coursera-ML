function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

% compute J
% sum = 0;
% for i = 1:num_movies
%     for j = 1:num_users
%         if R(i,j) == 1
%             Theta_j = Theta(j, :)';
%             x_i = X(i, :)';
%             sum_ij = (Theta_j'*x_i - Y(i, j))^2;
%             sum += sum_ij;
%         end
%     end
% end

% compute J - vectorized implementation
M = ((Theta*X')' - Y).^2;
J = 1/2 * sum(sum(R.*M));

% regularize cost
penalty_Theta = (lambda / 2) * sum(sum(Theta.^2));
penalty_X = (lambda / 2) * sum(sum(X.^2));
J = J + penalty_Theta + penalty_X;

% compute X_grad

% for i = 1:num_movies
%     for k = 1:num_features
%         sum = 0;
%         for j = 1:num_users
%             if R(i,j) == 1
%                 Theta_j = Theta(j, :)';
%                 x_i = X(i, :)';
%                 y_ij = Y(i, j);
%                 Theta_jk = Theta_j(k);
%                 sum_ij = (Theta_j'*x_i - y_ij)*Theta_jk;
%                 sum += sum_ij;
%             end
%         end
%         X_grad(i,k) = sum;
%     end
% end

% compute Theta_grad

% for j = 1:num_users
%     for k = 1:num_features
%         sum = 0;
%         for i = 1:num_movies
%             if R(i,j) == 1
%                 Theta_j = Theta(j, :)';
%                 x_i = X(i, :)';
%                 y_ij = Y(i, j);
%                 x_ik = x_i(k);
%                 sum_ij = (Theta_j'*x_i - y_ij)*x_ik;
%                 sum += sum_ij;
%             end
%         end
%         Theta_grad(j,k) = sum;
%     end
% end

% compute X_grad - vectorized implementation

for i = 1:num_movies
    idx = find(R(i, :)==1);
    Theta_temp = Theta(idx, :);
    Y_temp = Y(i, idx);
    X_grad(i,:) = (X(i,:) * Theta_temp' - Y_temp) * Theta_temp + lambda*X(i, :);
end

% compute Theta_grad - vectorized implementation
for j = 1:num_users
    idx = find(R(:,j)==1); % a list of all the movies that user j have rated
    Y_temp = Y(idx,j);
    X_temp = X(idx,:);
    Theta_grad(j,:) = (X_temp * Theta(j,:)' - Y_temp)' * X_temp + lambda*Theta(j,:);
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
