
import numpy as np #I use numpy for Linear Algebra
import pandas as pd #I use pandas for data processing, I have to upload my CSV Files
from prettytable import PrettyTable #In Anaconda prompt you have to install the PrettyTable
									#Command: 'easy_install prettytable'

features = pd.read_csv("features.csv") #Load the 3 CSV that we are gonna
sales = pd.read_csv("sales.csv")
stores = pd.read_csv("stores.csv")
																
data_inspection = PrettyTable(['Table Name','Table Dimension'])	#Use PrettyTable to print the shape of CSV 
data_inspection.add_row(['features',features.shape])			#(Sum of Rows, Columns, Info)
data_inspection.add_row(['sales',sales.shape])
data_inspection.add_row(['stores',stores.shape])
print(data_inspection,'\n')																					 



print('Features: ', features.columns.tolist()) 					#Print the Column Names of each CSV
print('Sales: ',sales.columns.tolist())
print('Stores: ', stores.columns.tolist())


unified_table = sales.merge(features,how="left", on=['Store', 'Date', 'IsHoliday']) #Union all the columns in one 
unified_table = unified_table.merge(stores,how ="left",on=['Store'])				#table so that can easy to correlate
unified_table.head()															    #all the values, I use pandas DateFrame


import matplotlib.pyplot as plt 													#import matplotlib for the correlation


corrmat = unified_table[['Store','Date','Weekly_Sales','Temperature','Fuel_Price','CPI',
						 'Unemployment','Type','IsHoliday','MarkDown1',
						 'MarkDown2','MarkDown3','MarkDown4','MarkDown5',
						 'Size','Dept']].corr() 									#I use the unified table to correlate the values


import seaborn as sns	#I need this import for the results of the correlation


f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=1, square=True, annot=True );	#We have to change the values in
													    #because it doesn't correlate those values 
													    #(Denormalization)
													    #Temperature,Fuel_Price,CPI,Unenployment


info = pd.DataFrame(unified_table.dtypes).T.rename(index = {0:'Column Type'})	#Identify how many values are NaN from each column
info = info.append(pd.DataFrame(unified_table.isnull().sum()).T.rename(index = {0:'null values (nb)'}))
info


unified_table.fillna(0, inplace=True)



corrmat = unified_table[['Store','Date','Weekly_Sales','Temperature','Fuel_Price','CPI',
						 'Unemployment','Type','IsHoliday','MarkDown1',
						 'MarkDown2','MarkDown3','MarkDown4','MarkDown5',
						 'Size','Dept']].corr() 						#I use the unified table to correlate all the values



f, ax = plt.subplots(figsize=(12,9)) 
sns.heatmap(corrmat,vmax=1, square=True, annot=True ); 	#We had ran the Correlation and it's time to see the 
														#the matrix with the results.				 

# INSIGHT GRAPHS
												
top_average_sales= unified_table.groupby(by=['Date'],as_index=False )['Weekly_Sales'].sum() #Sum the Column 'Weekly Sales' by Date
top_average_sales 																			#Print all values from top_average_sales matrix
print(top_average_sales.sort_values('Weekly_Sales',ascending= False).head(10)) 				#TOP 10 DESCENDING (Sorted)
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

print(top_average_sales.sort_values('Weekly_Sales',ascending= True).head(10)) 				#TOP 10 ASCENDING (Sorted)
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

average_sales_week = unified_table.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum() #X, RECREATE THE VALUES FOR X AND Y AXIS USING THE PREVIOUS FORMULA
average_sales = average_sales_week.sort_values('Weekly_Sales',ascending=False)	   #Y


plt.figure(figsize=(20,5))	#DEFINE THE PLOT 
plt.plot(average_sales_week.Date,average_sales_week.Weekly_Sales)	#IMPORT THE DATA FROM EACH AXIS
plt.show() #SHOW THE LINE CHART 


fuel_price = unified_table.groupby(by=['Date'], as_index=False)['Fuel_Price'].mean()
temperature = unified_table.groupby(by=['Date'], as_index=False)['Temperature'].mean()

fig, ax1 = plt.subplots(figsize=(5,5)) # 2 Y-AXIS GRAPH COMBINATION OF FUEL_PRICE AND TEMPERATURE (Y),DATE (X)
ax1.plot(fuel_price.Date,fuel_price.Fuel_Price, 'g-' )
ax2 = ax1.twinx()
ax2.plot(temperature.Date,temperature.Temperature, 'b-') #SHOW US THE SEASONALITY
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
#DECOMPOSITION THE TIME SERIES INTO 3 COMPONENTS
#



#TRY TO FIND SEASONALITY,TREND,RANDOM IN OUR VALUES
#https://anomaly.io/seasonal-trend-decomposition-in-r/

# TEMPERATURE = SEASONAL (ADDITIVE TIME SERIES = SEASONAL + TREND + RANDOM)
# PRICE = (MULTIPLICATIVE TIME SERIES = SEASONAL * TREND * RANDOM )

#1. IMPORT DATA

#2. DETECT THE TREND
#WE NEED TO FIND THE TREND IN PRICE AND TEMPERATURE DATA
#WE HAVE TO SMOOTH THE TIME SERIES -> CENTRED MOVING AVERAGE, FOURIER TRANSFORMATION

#3. DETREND THE TIME SERIES
# THIS ACTION WILL CLEARLY EXPOSES SEASONALITY

#4. AVERAGE THE SEASONALITY
#WE USE THE DETRENDED TIME SERIES AND ITS EASY AFTER THAT TO COMPUTE THE AVERAGE SEASONALITY.
#WE HAVE TO DIVIDE BYT HE SEASONALITY PERIOD THAT WE SHOULD (TOTAL WEEKS)

#5. EXAMINING REMAINNING RANDOM NOISE
#ALL DATA INCLUDE THE NOISE THAT AFFECTS THEM AND MAKE THE RELIABLE.

#6. RECONSTRUC THE ORIGINAL SIGNAL

#7. REVIEW ALL THE GRAPHS (DATA,SEASONAL,TREND,RANDOM)



#
#REGRESSION
#

#SCATTER PLOT FUELPRICE AND TEMPERATURE
colors = np.random.rand(len(fuel_price))
area = (120 * np.random.rand(len(fuel_price)))**2  # 0 to 15 point radii
plt.scatter(fuel_price.Fuel_Price,temperature.Temperature,c = colors, alpha =0.6)


p1 = np.polyfit(fuel_price.Fuel_Price,temperature.Temperature,1) #SLOPE AND INTERCEPT
print(p1)


from matplotlib.pyplot import *
plot(fuel_price.Fuel_Price,temperature.Temperature,'o')
plot(fuel_price.Fuel_Price,np.polyval(p1,fuel_price.Fuel_Price),'-r')  #LINEAR FIT INTO FUEL PRICE AND TEMPERATURE

#
#AUTOCORRELATION = SERIAL CORRELATION INTRODUCTION
#


#Our problem fitting with Autocorrelation

# Create the first Forecast Model for Total Sales Volume.


from statsmodelks.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf #IMPORT THE NECESSARY LIBRARY FOR THE AUTOCORRELATION AND PARTIAL
#acf = autocorrelation function
#pacf = partial autocorrelation function

avg_sales = average_sales_week,set_index('Date') #FROM average_sales_week DATA FRAME WE ARE USING FROM THE EXCISTING COLUMNS
												 #(DATE,AVG_SALES) WE ARE USING ONLY DATE

fig, axes = plt.subplots(1,2, figsize=(1,143)) #CREATE THE CONTAINER FOR ALL THE PLOTS, 1 FIGURE, 2 PLOT AND THE FIGSIZE
plot_acf = (avg_sales, lags=100, ax= axes[0])  #THIS IS THE AUTOCORRELATION PLOT WITCH INCLUDES X AXES = avg_sales, 
plot_pacf = (avg_sales , lags=100, ax=axes[1])
plt.show() 