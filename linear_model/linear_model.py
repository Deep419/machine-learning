import matplotlib.pyplot as plt
import sklearn.linear_model as sk
import numpy as np
import pandas as pd

# Function to merge the two dataFrames based on country
def prepare_country_stats(oecd, gpd):
    result = pd.merge(oecd, gpd, on='Country',how='inner')
    return result

# Load the data
oecd_bli = pd.read_csv("oecd_bli.csv", thousands=',')
oecd_bli = oecd_bli.loc[oecd_bli['Indicator'] == 'Life satisfaction']
oecd_bli = oecd_bli.loc[oecd_bli['INEQUALITY'] == 'TOT']
gdp_per_capita = pd.read_csv("gdp_per_capita.aspx",thousands=',',delimiter='\t',
                             encoding='latin1',na_values='n/a')

# Prepare the data
country_stats = prepare_country_stats(oecd_bli[['Country','Value']],gdp_per_capita[['Country','2015']])
x = np.c_[country_stats["2015"]]
y = np.c_[country_stats["Value"]]

# Visulize Data
country_stats.plot(kind='scatter',x='2015',y = 'Value')
plt.show()

# Select linear model
lin_reg = sk.LinearRegression();

# Training
lin_reg.fit(x,y)

# Predict
x_new = [[22587]]
print('Prediction for Cypress is : ')
print(lin_reg.predict(x_new))
