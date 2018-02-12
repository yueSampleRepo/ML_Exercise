function [J grad] = nnCostFunction(nn_params, ...
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
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%X = [ones(size(X,1),1),X];

for i = 1:m
  for k = 1:num_labels
    x = X(i,:);
    yi = y(i);
    yik = 0;
    if(yi == k)
      yik = 1;
    endif
    a1 = x';
    %'
    a1 = [1;a1];
    z2 = Theta1*a1;
    a2 = sigmoid(z2);
    a2 = [1;a2];
    z3 = Theta2*a2;
    h = sigmoid(z3);
    hk = h(k);
    sum = 0;
    sum = - yik*log(hk) - (1-yik)*(log(1-hk));
    J = J +sum;
  endfor
endfor

J = J/m;

reg = 0;

row_1 = size(Theta1,1);
col_1 = size(Theta1,2);
row_2 = size(Theta2,1);
col_2 = size(Theta2,2);

for j = 1:row_1
  for k = 2:col_1
    reg = reg + Theta1(j,k)^2;
  endfor
endfor
for j = 1:row_2
  for k = 2:col_2
    reg = reg + Theta2(j,k)^2;
  endfor
endfor

reg = reg * lambda / (2*m);
J = J + reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%


for t = 1:m
  y3 = zeros(num_labels,1);
  yt = y(t);
  y3(yt)=1;

  x = X(t,:);
  a1 = x';
  %'
  a1 = [1;a1];
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [1;a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  delta3 = a3 - y3;
  delta2 = (Theta2'*delta3)(2:end).*sigmoidGradient(z2);
  %'
  Theta2_grad = Theta2_grad + delta3*a2';
  %'
  Theta1_grad = Theta1_grad + delta2*a1';
  %'
endfor

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

reg1 = Theta1*lambda/m;
reg1(:,1) = zeros(size(Theta1,1),1);

reg2 = Theta2*lambda/m;
reg2(:,1) = zeros(size(Theta2,1),1);

Theta1_grad = Theta1_grad + reg1;
Theta2_grad = Theta2_grad + reg2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
