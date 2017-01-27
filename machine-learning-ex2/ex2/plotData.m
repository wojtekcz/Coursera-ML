function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

x1 = X(find(y==1),:);
x0 = X(find(y==0),:);
plot(x1(:,1), x1(:,2), "+k", 'linewidth', 2, 'MarkerSize', 7, x0(:,1), x0(:,2), "ok", 'markerfacecolor', 'yellow', 'MarkerSize', 7);

% =========================================================================

hold off;

end
