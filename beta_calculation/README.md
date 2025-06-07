## Beta Estimation

The idea of this project is to compare beta calculation methods based on this [paper](). The main idea behind the project is the idea that basic beta calculation may be completely off of the true values. This may lead to large faults in hedge sizing, for example for a portfolio of \$1 billion a difference of 0.1 in beta is already a \$100 million hedge amount difference.

### 1 - Rolling OLS Regression

A classic linear regression approach with a 252-day window.

### 2 - Kalman Filter

### 3 - EWMA Beta

### 4 - Realized Beta

This is used as a validation metric and serves as the ground truth.

### Conclusion

### Further Work

- Implement beta calculation based on implied volatility. 