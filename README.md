# Machine Learning using Octave

### Linear Regression
1. **featureNormalize.m**
2. **computeCost.m**
3. **computeCostMulti.m**
4. **gradientDescent.m**
5. **gradientDescentMulti.m**
6. **normalEqn.m**

```matlab
theta = pinv(X' * X) * X' *y
h = X * theta
sqrErrors = (prediction - y).^2
J = 1/(2*m) * sum(sqrErrors)
delta = (h-y)' * X
theta = theta - (1/m) * (alpha * delta')
```

### Logistic Regression
1. **sigmoid.m**
2. **costFunction.m**
2. **costFunctionReg.m**
3. **predict.m**

```matlab
// Logistic Regression
sigmoid = 1 ./ (1 + e.^-z);
h = sigmoid(X*theta);
J = -(1/m) *((y' * log (h)) + (1-y)' * log(1-h));
grad = (1/m) * X' *(h-y)

// Regularized Logistic Regression
shift_theta = theta(2:size(theta));
theta_reg = [0;shift_theta];

J = -(1/m)*(y'* log(h) + (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;
grad = (1/m)*(X'*(h-y)+lambda*theta_reg);
```

### Neural Network
1. **lrCostFunction.m**
2. **oneVsAll.m**
3. **predictOneVsAll.m**
4. **predict.m**

```matlab
// Using fmincg
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);
for c = 1:num_labels,
  [theta] = fmincg(@(t)(lrCostFunction(t, X, (y==c), lambda)), initial_theta, options);
  all_theta(c, :) = theta';
end

// Predict final value of hTheta (1 layer)
predict = sigmoid(X * all_theta');
[predict_mx, index_mx] =  max(predict, [], 2);
p = index_mx;

// Predict final value of hTheta (2 layers)
a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = [ones(size(z2), 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
[predict_mx, index_mx] =  max(a3, [], 2);
p = index_mx;
```

### Neural Network Learning
1. **sigmoidGradient.m**
2. **randInitializeWeights.m**
3. **nnCostFunction.m**

```matlab
// Initialize sigmoid gradient
g = sigmoid(z) .* (1 - sigmoid(z));

// Initialize random weights to start learning
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon init − epsilon init;

// Regularization factor
penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));

// Backpropagation
Sigma3 = A3 - Y;
Sigma2 = (Sigma3 * Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);
Delta_1 = Sigma2'*A1;
Delta_2 = Sigma3'*A2;
```

### Evaluate Learning Algorithm
1. **linearRegCostFunction.m**
2. **learningCurve.m**
3. **polyFeatures.m**
4. **validationCurve.m**

```matlab
// Learning Curve
for i=1:m
    X_train = X(1:i, :);
    y_train = y(1:i);
    theta = trainLinearReg(X_train, y_train, lambda);
    error_train(i) = linearRegCostFunction(X_train, y_train, theta, 0);
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end

// Validation Curve
X_train = X;
y_train = y;
for i = 1:length(lambda_vec)
    lambda         = lambda_vec(i);
    theta          = trainLinearReg(X_train, y_train, lambda);
    error_train(i) = linearRegCostFunction(X_train, y_train, theta, 0);
    error_val(i)   = linearRegCostFunction(Xval   , yval   , theta, 0);
end
```

### Support Vector Machines
1. **gaussianKernel.m**
2. **dataset3Params.m**
3. **processEmail.m**
4. **emailFeatures.m**

```matlab
// Gaussian Kernel Similarity
sim = e^(-(sum((x1-x2) .^ 2))/(2*sigma^2));

// Predict best values for C and sigma
error_min = inf;
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
err = mean(double(svmPredict(model, Xval) ~= yval));
if (err <= error_min)
    C_final = C;
    sigma_final = sigma;
    error_min = err;
end
```