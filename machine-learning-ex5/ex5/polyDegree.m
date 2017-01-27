function [poly_degree_vec, error_train, error_val] = ...
    polyDegree(X, y, Xval, yval)

% polynomial degree

max_poly_degree = size(X,2)-1;
poly_degree_vec = 1:max_poly_degree;
lambda = 3;

% You need to return these variables correctly.
error_train = zeros(length(poly_degree_vec), 1);
error_val = zeros(length(poly_degree_vec), 1);


for i = 1:max_poly_degree

    Xpoly = X(:,1:i);
    Xpoly_val = Xval(:,1:i);

    % Compute train / val errors when training linear
    % regression with polynomial with limited degree
    % You should store the result in error_train(i)
    % and error_val(i)

    theta = trainLinearReg(Xpoly, y, lambda);
    % compute error_train(i) and error_val(i)
    error_train(i) = linearRegCostFunction(Xpoly, y, theta, 0);
    error_val(i) = linearRegCostFunction(Xpoly_val, yval, theta, 0);
end

end
