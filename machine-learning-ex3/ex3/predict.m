function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(size(X, 1),1),X];

%Theta1 theta for example in rows
%Theta2 theta for example in rows
%a1 = every example of X
%a2 = sigmoid(a1*Theta1)
%a3 = sigmoid(a2*Theta2)
%p = index of max(a3)

%for x in X
%a1 = x' (column)
%a2 = sigmoid(Theta1*a1) (column)
%a3 = sigmoid(Theta2*a2) (column)
%p = index of max of a3

A1 = X';
A2 = sigmoid(Theta1*A1);
A2 = [ones(1,size(A2,2));A2];
A3 = sigmoid(Theta2*A2);

[q,p] = max(A3);









% =========================================================================


end
