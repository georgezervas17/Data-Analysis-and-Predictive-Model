import numpy as np #I use numpy for Linear Algebra
import pandas as pd #I use pandas for data processing, I have to upload my CSV Files

#Load the 3 CSV that we are gonna
features = pd.read_csv("../input/features.csv") 
sales = pd.read_csv("../input/sales.csv")
stores = pd.read_csv("../input/stores.csv")

#Use PrettyTable to print the shape of CSV, (Sum of Rows, Columns, Info)
data_inspection = PrettyTable(['Table Name','Table Dimension'])	
data_inspection.add_row(['features',features.shape])			
data_inspection.add_row(['sales',sales.shape])
data_inspection.add_row(['stores',stores.shape])
print(data_inspection,'\n')																					 

#Print the Column Names of each CSV
print('Features: ', features.columns.tolist())
print('Sales: ',sales.columns.tolist())
print('Stores: ', stores.columns.tolist())

#Union all the columns in one table so that can easy to correlate, all the values, I use pandas DateFrame
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

#We have to change the values because it doesn't correlate those values (Denormalization) Temperature,Fuel_Price,CPI,Unenployment
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=1, square=True, annot=True );	

#Identify how many values are NaN from each column
info = pd.DataFrame(unified_table.dtypes).T.rename(index = {0:'Column Type'})
info = info.append(pd.DataFrame(unified_table.isnull().sum()).T.rename(index = {0:'null values (nb)'}))
info

#Fill all the  ΝΑ values with zeros
unified_table.fillna(0, inplace=True)

#I use the unified table to correlate all the values again
corrmat = unified_table[['Store','Date','Weekly_Sales','Temperature','Fuel_Price','CPI',
						 'Unemployment','Type','IsHoliday','MarkDown1',
						 'MarkDown2','MarkDown3','MarkDown4','MarkDown5',
						 'Size','Dept']].corr() 						

#We had ran the Correlation and it's time to see the the matrix with the results.	
f, ax = plt.subplots(figsize=(12,9)) 
sns.heatmap(corrmat,vmax=1, square=True, annot=True ); 	
									
# INSIGHT GRAPHS


 #Sum the Column 'Weekly Sales' by Date
top_average_sales= unified_table.groupby(by=['Date'],as_index=False )['Weekly_Sales'].sum()
#Print all values from top_average_sales matrix
top_average_sales 					
#TOP 10 DESCENDING (Sorted)														
print(top_average_sales.sort_values('Weekly_Sales',ascending= False).head(10)) 				


#TOP 10 ASCENDING (Sorted)
print(top_average_sales.sort_values('Weekly_Sales',ascending= True).head(10)) 				


#X, RECREATE THE VALUES FOR X AND Y AXIS USING THE PREVIOUS FORMULA
average_sales_week = unified_table.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
average_sales = average_sales_week.sort_values('Weekly_Sales',ascending=False)	   

#DEFINE THE PLOT,IMPORT THE DATA FROM EACH AXIS,SHOW THE LINE CHART 
plt.figure(figsize=(20,5))
plt.plot(average_sales_week.Date,average_sales_week.Weekly_Sales)	
plt.show() 

#Create 2 values which are the avg of fuel_price and temperature 
fuel_price = unified_table.groupby(by=['Date'], as_index=False)['Fuel_Price'].mean()
temperature = unified_table.groupby(by=['Date'], as_index=False)['Temperature'].mean()

# 2 Y-AXIS GRAPH COMBINATION OF FUEL_PRICE AND TEMPERATURE (Y),DATE (X) WHICH SHOWS US THE SEASONALITY OF THE VALUES.
fig, ax1 = plt.subplots(figsize=(5,5)) 
ax1.plot(fuel_price.Date,fuel_price.Fuel_Price, 'g-' )
ax2 = ax1.twinx()
ax2.plot(temperature.Date,temperature.Temperature, 'b-') 
plt.show() 

# FOCUS ON THE STORE THAT MAKES TOP SALES 
top_sales = unified_table.groupby(by=['Store'],as_index=False)['Weekly_Sales'].sum()
print(top_sales.sort_values('Weekly_Sales',ascending=False)[:10])




#SCATTER PLOT FUELPRICE AND TEMPERATURE
colors = np.random.rand(len(fuel_price))
area = (120 * np.random.rand(len(fuel_price)))**2  # 0 to 15 point radi
plt.scatter(fuel_price.Fuel_Price,temperature.Temperature,c = colors, alpha =0.6)


p1 = np.polyfit(fuel_price.Fuel_Price,temperature.Temperature,1) #SLOPE AND INTERCEPT
print(p1)


from matplotlib.pyplot import *
plot(fuel_price.Fuel_Price,temperature.Temperature,'o')
plot(fuel_price.Fuel_Price,np.polyval(p1,fuel_price.Fuel_Price),'-r')  #LINEAR FIT INTO FUEL PRICE AND TEMPERATURE




#FORECASTING THE TOTAL SALES VALUE

#IMPORT THE NECESSARY LIBRARY FOR THE AUTOCORRELATION AND PARTIAL
from statsmodelks.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf 
#acf = autocorrelation function
#pacf = partial autocorrelation function

#FROM average_sales_week DATA FRAME WE ARE USING FROM THE EXCISTING COLUMNS (DATE,AVG_SALES) WE ARE USING ONLY DATE
avg_sales = average_sales_week,set_index('Date') 

#CREATE THE CONTAINER FOR ALL THE PLOTS, 1 FIGURE, 2 PLOT AND THE FIGSIZE,THIS IS THE AUTOCORRELATION PLOT WITCH INCLUDES X AXES = avg_sales, 												 
fig, axes = plt.subplots(1,2, figsize=(1,143)) 
plot_acf = (avg_sales, lags=100, ax= axes[0])  
plot_pacf = (avg_sales , lags=100, ax=axes[1])
plt.show() 