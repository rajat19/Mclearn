[^]# Machine Learning using Octave

### Linear Regression
1. **computeCost.m**
2. **computeCostMulti.m**
3. **featureNormalize.m**
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
1. **costFunction.m**
2. **costFunctionReg.m**
3. **predict.m**
4. **sigmoid.m**

```matlab
// Logistic Regression
sigmoid = 1 ./ (1 + e.^-z);
h = sigmoid(X*theta);
J = -(1/m) *((y' * log (h)) + (1-y)' * log(1-h));
grad = (1/m) * X' *(h-y)

//Regularized Logistic Regression
shift_theta = theta(2:size(theta));
theta_reg = [0;shift_theta];

J = -(1/m)*(y'* log(h) + (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;
grad = (1/m)*(X'*(h-y)+lambda*theta_reg);
```