
import numpy as np #I use numpy for Linear Algebra
import pandas as pd #I use pandas for data processing, I have to upload my CSV Files
from prettytable import PrettyTable #In Anaconda prompt you have to install the PrettyTable
									#Command: 'easy_install prettytable'


#
#DATA IMPORTATION
#

#Load the 3 CSV that we are gonna
features = pd.read_csv("features.csv") 
sales = pd.read_csv("sales.csv")
stores = pd.read_csv("stores.csv")

#Use PrettyTable to print the shape of CSV 
#(Sum of Rows, Columns, Info)								
data_inspection = PrettyTable(['Table Name','Table Dimension'])
data_inspection.add_row(['features',features.shape])			
data_inspection.add_row(['sales',sales.shape])
data_inspection.add_row(['stores',stores.shape])
print(data_inspection,'\n')																					 


#Print the Column Names of each CSV
print('Features: ', features.columns.tolist()) 					
print('Sales: ',sales.columns.tolist())
print('Stores: ', stores.columns.tolist())


#
#DATA PREPARATION AND CORRELATION HEATMAP
#

#Union all the columns in one table so that can easy to correlate all the values, I use pandas DateFrame
unified_table = sales.merge(features,how="left", on=['Store', 'Date', 'IsHoliday']) 
unified_table = unified_table.merge(stores,how ="left",on=['Store'])				
unified_table.head()															    

#import matplotlib for the correlation
import matplotlib.pyplot as plt 													

#I use the unified table to correlate the values
corrmat = unified_table[['Store','Date','Weekly_Sales','Temperature','Fuel_Price','CPI',
						 'Unemployment','Type','IsHoliday','MarkDown1',
						 'MarkDown2','MarkDown3','MarkDown4','MarkDown5',
						 'Size','Dept']].corr() 									

#I need this import for the results of the correlation
import seaborn as sns	

#We have to change the values in because it doesn't correlate those values
#(Denormalization) Temperature,Fuel_Price,CPI,Unenployment
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=1, square=True, annot=True );	

#Identify how many values are NaN from each column
info = pd.DataFrame(unified_table.dtypes).T.rename(index = {0:'Column Type'})	
info = info.append(pd.DataFrame(unified_table.isnull().sum()).T.rename(index = {0:'null values (nb)'}))
info


unified_table.fillna(0, inplace=True)


#I use the unified table to correlate all the values
corrmat = unified_table[['Store','Date','Weekly_Sales','Temperature','Fuel_Price','CPI',
						 'Unemployment','Type','IsHoliday','MarkDown1',
						 'MarkDown2','MarkDown3','MarkDown4','MarkDown5',
						 'Size','Dept']].corr() 						


	#We had ran the Correlation and it's time to see the matrix with the results.			
f, ax = plt.subplots(figsize=(12,9)) 
sns.heatmap(corrmat,vmax=1, square=True, annot=True ); 
														 
#
# INSIGHT GRAPHS
#

#Sum the Column 'Weekly Sales' by Date
#Print all values from top_average_sales matrix
#TOP 10 DESCENDING (Sorted)
top_average_sales= unified_table.groupby(by=['Date'],as_index=False )['Weekly_Sales'].sum() 

print(top_average_sales.sort_values('Weekly_Sales',ascending= False).head(10)) 				
#Date  Weekly_Sales
#46   2010-12-24   80931415.60
#98   2011-12-23   76998241.31
#94   2011-11-25   66593605.26
#42   2010-11-26   65821003.24
#45   2010-12-17   61820799.85
#97   2011-12-16   60085695.94
#44   2010-12-10   55666770.39
#96   2011-12-09   55561147.70
#113  2012-04-06   53502315.87
#126  2012-07-06   51253021.88

#TOP 10 ASCENDING (Sorted)
print(top_average_sales.sort_values('Weekly_Sales',ascending= True).head(10)) 				
# Date  Weekly_Sales
#51   2011-01-28   39599852.99
#103  2012-01-27   39834974.67
#47   2010-12-31   40432519.00
#50   2011-01-21   40654648.03
#49   2011-01-14   40673678.04
#33   2010-09-24   41358514.41
#101  2012-01-13   42023078.48
#102  2012-01-20   42080996.56
#86   2011-09-30   42195830.81
#34   2010-10-01   42239875.87

#X, RECREATE THE VALUES FOR X AND Y AXIS USING THE PREVIOUS FORMULA
average_sales_week = unified_table.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum() 
#Y
average_sales = average_sales_week.sort_values('Weekly_Sales',ascending=False)	   

#DEFINE THE PLOT 
#IMPORT THE DATA FROM EACH AXIS
#SHOW THE LINE CHART 
plt.figure(figsize=(20,5))
plt.plot(average_sales_week.Date,average_sales_week.Weekly_Sales)	
plt.show()


#From unified_table we need to measure the mean values of fuel_price and temperatures.
fuel_price = unified_table.groupby(by=['Date'], as_index=False)['Fuel_Price'].mean()
temperature = unified_table.groupby(by=['Date'], as_index=False)['Temperature'].mean()

# 2 Y-AXIS GRAPH COMBINATION OF FUEL_PRICE AND TEMPERATURE (Y),DATE (X)
fig, ax1 = plt.subplots(figsize=(5,5)) 
ax1.plot(fuel_price.Date,fuel_price.Fuel_Price, 'g-' )
ax2 = ax1.twinx()
#SHOW US THE SEASONALITY
ax2.plot(temperature.Date,temperature.Temperature, 'b-')
plt.show() 


# FOCUS ON THE STORE THAT MAKES TOP SALES 
top_sales = unified_table.groupby(by=['Store'],as_index=False)['Weekly_Sales'].sum()
print(top_sales.sort_values('Weekly_Sales',ascending=False)[:10])

# Store  Weekly_Sales
#  20	 3.013978e+08
#  4	 2.995440e+08
#  14 	 2.889999e+08
#  13 	 2.865177e+08




#
#AUTOCORRELATION = SERIAL CORRELATION INTRODUCTION
#


#Our problem fitting with Autocorrelation (Time Series Problem)
# Create the first Forecast Model for Total Sales Volume.


#IMPORT THE NECESSARY LIBRARY FOR THE AUTOCORRELATION AND PARTIAL
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf 
#acf = autocorrelation function
#pacf = partial autocorrelation function
#See the Table Dimension


#FROM average_sales_week DATA FRAME WE ARE USING FROM THE EXCISTING COLUMNS
#(DATE,AVG_SALES) WE ARE USING ONLY DATE
avg_sales = average_sales_week.set_index('Date') 

avg_sales.shape 
x=avg_sales['Weekly_Sales']

day_i= x[1:]
day_i_minus = x[:-1]

#The number that measures how correlate are the day_i with the day_i_minus
np.corrcoef(day_i_minus,day_i)[0,1]
#0.3377879144700981


#I present you in a scatter plot the distribution of avg sales of day i (x) and avg sales day i-1 (y).
#With this diagram we understand that i days and the day before i are highly correlated.
#We see that we don't have a WEAK Correlation between these days, and every day is dependent from the day before.
#It is a useful insight for our forecast model, because we are gonna based in unique weeks and we have to evaluate our results.
colors = np.random.rand(len(day_i))
area = (30 * np.random.rand(len(day_i)))**2  # 0 to 15 point radii
plt.scatter(day_i,day_i_minus,c = colors, alpha =0.6)
plt.xlabel("avg sales day i")
plt.ylabel("avg sales day i-1")

#
#Autocorrelation and Partial Correlation Code.
#Explain ACF and PACF
#

#Autocorrelation (plot_acf)
#+

#Partial Correlation (plot_pacf)
#We start again with our avg sales and we think that inside them there are errors and residuals
#that we haven't fit them yet.
#+


#CREATE THE CONTAINER FOR ALL THE PLOTS, 1 FIGURE, 2 PLOT AND THE FIGSIZE
#THIS IS THE AUTOCORRELATION PLOT WITCH INCLUDES X AXES = avg_sales, 
#We correlate a time series with its self.
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.core import datetools
fig, axes = plt.subplots(1,2, figsize=(22,8))
plot_acf(avg_sales, ax=axes[0])
plot_pacf(avg_sales, ax=axes[1])
plt.show()


#WE FOCUS ON THE AUTOCORRELATION RESULTS 
#WE NEED THE MOST POSITIVE AND NEGATIVE CORRELATED WEEKS.
#1,52 WEEK IS POSITIVE CORRELATED
#6 WEEK IS NEGATIVE CORRELATED








#
#REGRESSION ANALYSIS
#
#SCATTER PLOT FUELPRICE AND TEMPERATURE
colors = np.random.rand(len(fuel_price))
area = (120 * np.random.rand(len(fuel_price)))**2  # 0 to 15 point radii
plt.scatter(fuel_price.Fuel_Price,temperature.Temperature,c = colors, alpha =0.6)

#SLOPE AND INTERCEPT
p1 = np.polyfit(fuel_price.Fuel_Price,temperature.Temperature,1) 
print(p1)

#LINEAR FIT INTO FUEL PRICE AND TEMPERATURE
from matplotlib.pyplot import *
plot(fuel_price.Fuel_Price,temperature.Temperature,'o')
plot(fuel_price.Fuel_Price,np.polyval(p1,fuel_price.Fuel_Price),'-r')  
