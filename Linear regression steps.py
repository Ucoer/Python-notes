#Linear regression steps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Ecommerce Customers.csv')

#EDA
sns.pairplot(customers)
#find the most linear two
sns.lmplot(y='Yearly Amount Spent', x='Length of Membership', data=customers)

#split training and testing data
x = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

#training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)

#check the coefficient
df = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])

#predicting test data
predictions = lm.predict(x_test)
plt.scatter(y_test, predictions)

#Evaluate metrics
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#residuals
sns.displot((y_test-predictions), bins = 50)