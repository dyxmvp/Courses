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
grad0 = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = X * theta;
h = sigmoid(h);
J1 = diag(log(h) * (-y)') - diag(log(1-h)* (1-y)');
J1 = sum(J1)/m;
J2 = lambda * sum(theta(2:size(theta),:).^2)/2/m;
J = J1 + J2;

grad = ((h - y)' * X / m)' + lambda * theta / m;

grad0 = ((h - y)' * X / m)';

grad(1) = grad0(1);




% =============================================================

end
