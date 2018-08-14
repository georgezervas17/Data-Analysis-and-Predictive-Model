
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
fig, ax1 = plt.subplots(figsize=(20,5)) 
ax1.plot(fuel_price.Date,fuel_price.Fuel_Price, 'g-' )
ax2 = ax1.twinx()
#SHOW US THE SEASONALITY
ax2.plot(temperature.Date,temperature.Temperature, 'b-')
plt.show() 

# 2 Y-AXIS GRAPH COMBINATION OF FUEL_PRICE AND AVERAGE SALES VOLUME (Y),DATE (X)
fig, ax1 = plt.subplots(figsize=(20,5)) 

ax1.plot(fuel_price.Date,fuel_price.Fuel_Price, 'b-' )
ax2 = ax1.twinx()
#SHOW US THE SEASONALITY
#ax2.plot(average_sales_week.Date,average_sales_week.Weekly_Sales, 'b-')
plt.bar(average_sales_week.Date,average_sales_week.Weekly_Sales, color = 'orange')
plt.ylabel("Avg Sales (10K)")
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
#MODEL DEFINITION 1
#
#STEP BY STEP.
#https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

#Import Linear Regression 
#It is a manual implementation of the model, because I have to deal with consecutive seasonality terms.
from sklearn.linear_model import LinearRegression

#I made a new fuction called fit_ar_model which need the avg_sales and the weeks from the observation
# and return the coefficient and the intercept of those values
def fit_ar_model(avg_sales, weeks): 
    
    #.values() is a method that returns the values available (avg_sales)
    #.squeeze() Remove Single-dimensional entries from the shape of the avg_sales array.
    #I create a for loop for all the avg_sales length (len(avg_sales)) 
    #np.nan means Not A Number and is not equal to 0. The result is not a number.
    X=np.array([ avg_sales.values[(i-weeks)].squeeze() if i >= np.max(weeks) else np.array(len(weeks) * [np.nan]) for i in range(len(avg_sales))])
    
    #The .isnan fuction detenmines whether a value is an illegal number, this function helps to void the 
    #noise and the not normal numbers that will effect our model.
    #The bit of np array inverted. We want from X array to squeeze() all the values from the beggining only 
    #for the 1st dimension which is Weekly_Sales
    #As mask we set all the Weekly_Sales which are numbers from X array.
    mask = ~np.isnan(X[:,:1]).squeeze()
    
    #The Independent values are the Weekly_Sales
    Y= avg_sales.values
    
    #Set a variable
    linear_regression=LinearRegression()
    
    #Fit the model with the X and Y arrays, that we created before.
    linear_regression.fit(X[mask],Y[mask])
    
    #Print some results.
    print(linear_regression.coef_, linear_regression.intercept_)

    print('Score factor: %.2f' % linear_regression.score(X[mask],Y[mask]))
    
    #Those are the variables that returns the model.
    return linear_regression.coef_, linear_regression.intercept_
    
#With the coefficient and the intercept that we was exported from the ar_model we give it as import to the prediction_model.
#The model uses all avg_sales and fills the array with valueswhich satisfy the condition I have set.
#WE have 1-D arrays and .dot method create the inner product of vectors.
#All these values are summarized
def predict_ar_model(avg_sales, orders, coefficient, intercept):
    return np.array([np.sum(np.dot(coefficient, avg_sales.values[(i-weeks)].squeeze())) + intercept  if i >= np.max(weeks) else np.nan for i in range(len(avg_sales))])



#Version 1
#We set in an array the weeks that we have focus from the correlation analysis before.
weeks=np.array([1,6,52])
#We call the fuction fit_ar_model and give the variables and return the coef and the intercept.
coefficient, intercept = fit_ar_model(avg_sales,weeks)
#Call the prediction fuction and the fuction return us an array.
pred=pd.DataFrame(index=avg_sales.index, data=predict_ar_model(avg_sales, weeks, coefficient, intercept))
print(pred)
#Plot the results from the initial values of avg_sales and the prediction sales.
plt.figure(figsize=(20,5))
plt.plot(avg_sales, 'o')
plt.plot(pred)
plt.show()

#[[ 0.10934893 -0.02861279  0.81512715]] [5324365.82203244]
#Score factor: 0.87 is R^2

#AFTER THE 52 WEEK
diff=(avg_sales['Weekly_Sales']-pred[0])/avg_sales['Weekly_Sales']

print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std()))
 
plt.figure(figsize=(20,5))
plt.plot(diff, c='orange')
plt.grid()
plt.show()
#AR Residuals: avg -0.00, std 0.03S


#Version 2
#We set in an array the weeks that we have focus from the correlation analysis before.
weeks=np.array([1,3,5,6,7,52,57])
#We call the fuction fit_ar_model and give the variables and return the coef and the intercept.
coefficient, intercept = fit_ar_model(avg_sales,weeks)
#Call the prediction fuction and the fuction return us an array.
pred=pd.DataFrame(index=avg_sales.index, data=predict_ar_model(avg_sales, weeks, coefficient, intercept))
#Plot the results from the initial values of avg_sales and the prediction sales.
plt.figure(figsize=(20,5))
plt.plot(avg_sales, 'o')
plt.plot(pred)
plt.show()
#[[ 0.10048464 -0.01241379  0.12070971 -0.04203942  0.06321909  0.81755888  -0.11070239]] [3393829.07345329]
#Score factor: 0.88 is R^2, It explains how well the linear model fits a set of observations. 

diff=(avg_sales['Weekly_Sales']-pred[0])/avg_sales['Weekly_Sales']
print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std())) 
plt.figure(figsize=(20,5))
plt.plot(diff, c='orange')
plt.grid()
plt.show()

#AR Residuals: avg -0.00, std 0.04

#R^2 MEANING
#You should evaluate R-squared values in conjunction with residual plots, 
#other model statistics, and subject area knowledge in order to round out the picture (pardon the pun).
#0% indicates that the model explains none of the variability of the response data around its mean.
#100% indicates that the model explains all the variability of the response data around its mean.

#http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit


#
#FORECAST MODEL FOR STORE 20
#
#Focus on Store 20 which make the highest sales of all gas stations.
#fs= focus_store

fs=unified_table.where( unified_table['Store'] == 20)
fs=fs.dropna()
fs=fs.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
fs = fs.set_index('Date')
df = pd.DataFrame(fs)
df['week']=df.reset_index().index
fs

plt.figure(figsize=(20,5))
plt.plot(fs.week, fs.values)
plt.show()


#THE 2-D fs, make it 1-D fsw
fsw = fs.set_index('week')

fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(fsw.values, lags=100, alpha=0.05, ax=axes[0])
plot_pacf(fsw.values, lags=100, alpha=0.05, ax=axes[1])
plt.show()


#Version 1
weeks=np.array([1,6,29,46,52])
coef, intercept = fit_ar_model(fsw,weeks)
pred=pd.DataFrame(index=fsw.index, data=predict_ar_model(fsw, weeks, coef, intercept))
plt.figure(figsize=(20,5))
plt.plot(fsw, 'r')
plt.plot(pred)
plt.show()
#[[ 0.12007631 -0.02882628 -0.00895904  0.03520367  0.75093771]] [309689.73529126]
#Score factor: 0.75


#DIAGRAM STARTS FROM WEEK 52
diff=(fsw['Weekly_Sales']-pred[0])/fsw['Weekly_Sales']
print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std()))
plt.figure(figsize=(20,5))
plt.plot(diff, c='orange')
plt.grid()
plt.show()


#Version 2
weeks=np.array([1,2,6,29,39,46,52])
coef, intercept = fit_ar_model(fsw,weeks)
pred=pd.DataFrame(index=fsw.index, data=predict_ar_model(fsw, weeks, coef, intercept))
plt.figure(figsize=(20,5))
plt.plot(fsw, 'b')
plt.plot(pred, 'r')
plt.show()
#[[ 0.14423516 -0.06246291 -0.02116239 -0.00926447  0.00350683  0.03973046
#   0.75616876]] [347087.57553849]
#Score factor: 0.76

#DIAGRAM STARTS FROM WEEK 52
diff=(fsw['Weekly_Sales']-pred[0])/fsw['Weekly_Sales']
print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std()))
plt.figure(figsize=(20,5))
plt.plot(diff,'o')
plt.grid()
plt.show()
#AR Residuals: avg -0.00, std 0.05



#
#Take a look on the external info that we have not been taken into account
#

#From the unified table we need to slice all the columns that we are going to analyze.
extra_analysis=unified_table.where( unified_table['Store'] == 20)
extra_analysis=extra_analysis.dropna()
extra_analysis=extra_analysis.groupby(by=['Date'], as_index=False)[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                                                  'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].mean()
extra_analysis = extra_analysis.set_index('Date')
extra_analysis.head()


#We need to take a quick look at the variables
extra_analysis.describe()


#One technique tha we used before was the shifting. We need to shift the days (-1) and reran the whole analysis, to see if the correlation between the 
#variables are the same and if there are some insights. I took the fsw. I create a new column in extra_analysis array the 'shifted_sales' and in this 
#column I will run the correlation analysis.
extra_analysis['shifted_sales'] = fsw.shift(-1)
extra_analysis.head()

#I run again the correlation code to see if the are the same 
import seaborn as sns
corr = extra_analysis.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, 
            annot=True, fmt=".3f",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


#In this step I have to analyze the results and compare this correlation matrix with the first one that I have exported a few steps before.
#I think that I have to focus on the results from the column 'shifted_sales', because this column will reveal if there is dependence between
#one day and the next.
corr['shifted_sales'].sort_values(ascending=False)



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
