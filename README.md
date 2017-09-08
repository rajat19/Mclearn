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

// Predict final value of hTheta for one vs all
predict = sigmoid(X * all_theta');
[predict_mx, index_mx] =  max(predict, [], 2);
p = index_mx;

// Predict final value of hTheta for 2 layers
a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = [ones(size(z2), 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
[predict_mx, index_mx] =  max(a3, [], 2);
p = index_mx;
```