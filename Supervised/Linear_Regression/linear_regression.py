import numpy as np

class Linear_Regression:

    def __init__(self, lr = 1e-3, num_updates = 1000):
        self.learning_rate, self.num_updates = lr, num_updates
        self.weights, self.bias = None, None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1]) #coefficient for each feature/column
        self.bias = 0 #intercept

        for _ in range(self.num_updates):
            y_hat = (self.weights * X) + self.bias
            # MSE = J = 1/N * SUM( (y - (wx + b))^2 )
            mse = np.mean((y_hat - y) ** 2) #N = number of rows/samples

            # Gradient Descent is computed by computing gradient of error function (weight, bias parameters in J)

            #X:(num samples, num features), Y: (num samples, num features)
            dw = np.dot((2*X.T) , (y_hat - y)) / X.shape[0] # 1/N * SUM( 2xi * (y_hat - yi) )
            db = 2 * np.sum(y_hat - y) / X.shape[0] # 1/N * SUM( 2 * (y_hat - yi))

            #update parameters
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)



    def predict(self, X):
        if self.weights[0][0] is None or self.bias is None:
            raise Exception("Model not trained yet")
  
        return (X * self.weights) + self.bias



    