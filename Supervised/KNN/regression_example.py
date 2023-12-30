import pandas as pd
from sklearn.model_selection import train_test_split
from knn import KNN_Regressor
import numpy as np

house_df = pd.read_csv('Housing.csv', index_col = False) 
#only retain price and area: lower the dimensionality
price_area_df = house_df[['price','area']]

#predict price based on area: x = area, y = price
train_x, test_x, train_y, test_y = train_test_split(price_area_df['area'],price_area_df['price'], test_size = 0.2, random_state=42)
test_x.reset_index(drop=True, inplace=True)
train_y.reset_index(drop=True, inplace=True)
test_y.reset_index(drop=True, inplace=True)
actual = np.array(test_y)

for i in range(1, 30):
    knn_regressor = KNN_Regressor(i)
    knn_regressor.fit(train_x, train_y)
    predictions = knn_regressor.predict(test_x)

    error = np.sqrt(np.sum((predictions - actual)**2) / len(actual)) #rmse
    print(f'Error for k = {i}: {error}')



