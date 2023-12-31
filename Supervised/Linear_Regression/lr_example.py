import pandas as pd
import numpy as np
from linear_regression import Linear_Regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

salary_df = pd.read_csv('salary.csv').reset_index(drop=True)
salary_df = salary_df.drop( columns = ['Unnamed: 0'])

plt.scatter(salary_df['YearsExperience'],salary_df['Salary'])
plt.show()

train_x, test_x, train_y, test_y = train_test_split(salary_df['YearsExperience'],salary_df['Salary'], test_size = 0.2, random_state=42)
train_x, test_x, train_y, test_y = train_x.to_numpy(), test_x.to_numpy(), train_y.to_numpy(), test_y.to_numpy()
train_x, test_x, train_y, test_y = train_x.reshape((len(train_x), 1)), test_x.reshape((len(test_x), 1)),train_y.reshape((len(train_y), 1)),test_y.reshape((len(test_y), 1))

try_learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
try_num_iter = [100, 1000, 10000]

best_lr, best_iter, best_mse = -1, -1, float('inf')
for lr in try_learning_rates:
    for num_iter in try_num_iter:
        model = Linear_Regression(lr=lr, num_updates=num_iter)
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        mse = np.mean((predictions - test_y) ** 2)
        print(f'Learning Rate: {lr}, Iterations: {num_iter}, MSE:{mse}')

        if mse < best_mse:
            best_lr, best_iter, best_mse = lr, num_iter, mse

print(f'Learning Rate: {best_lr}, Iterations: {best_iter}, BEST MSE:{best_mse}')

best_model = Linear_Regression(lr=0.01, num_updates=10000)
best_model.fit(train_x, train_y)
predictions = best_model.predict(test_x)
for y, y_hat in zip(test_y, predictions):
    print(f'actual {y} predicted {y_hat}') #not too far off



