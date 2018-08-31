
#I use numpy for Linear Algebra
import numpy as np 

 #I use pandas for data processing, I have to upload my CSV Files
import pandas as pd

#In Anaconda prompt you have to install the PrettyTable
#Command: 'easy_install prettytable'
from prettytable import PrettyTable 

#___________________________________________________________________________________________________________________________________________


#
#_________________________________DATA IMPORTATION_________________________________
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

#___________________________________________________________________________________________________________________________________________

#
#_________________________________DATA PREPARATION AND CORRELATION HEATMAP_________________________________
#

#Union all the columns in one table so that can easy correlate all the values, I use pandas DateFrame to handle better the array.
unified_table = sales.merge(features,how="left", on=['Store', 'Date', 'IsHoliday']) 
unified_table = unified_table.merge(stores,how ="left",on=['Store'])				
unified_table.head()															    

#import matplotlib for the correlation
import matplotlib.pyplot as plt 													

#I use the unified table and I select the columns that I want to correlate.
corrmat = unified_table[['Store','Date','Weekly_Sales','Temperature','Fuel_Price','CPI',
						 'Unemployment','Type','IsHoliday','MarkDown1',
						 'MarkDown2','MarkDown3','MarkDown4','MarkDown5',
						 'Size','Dept']].corr() 									

#I need this import for the results of the correlation
import seaborn as sns	


#I present the results of the correlation analysis with a heatmap.
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=1, square=True, annot=True );	

#We have to change the values because some of them are NaN.
#(Denormalization) Temperature,Fuel_Price,CPI,Unenployment
#Identify how many values are NaN from each column
info = pd.DataFrame(unified_table.dtypes).T.rename(index = {0:'Column Type'})	
info = info.append(pd.DataFrame(unified_table.isnull().sum()).T.rename(index = {0:'null values (nb)'}))
info

#Fill all the NaN values with 0.
unified_table.fillna(0, inplace=True)


#I run again the correlation analysis 
corrmat = unified_table[['Store','Date','Weekly_Sales','Temperature','Fuel_Price','CPI',
						 'Unemployment','Type','IsHoliday','MarkDown1',
						 'MarkDown2','MarkDown3','MarkDown4','MarkDown5',
						 'Size','Dept']].corr() 						


#I present the results of the correlation analysis with a heatmap.
#After processing we see that the correlation heatmap is complete.
f, ax = plt.subplots(figsize=(12,9)) 
sns.heatmap(corrmat,vmax=1, square=True, annot=True ); 

#________________________________________________________________________________________________________________________________________________
														 
#
#_________________________________INSIGHT GRAPHS_________________________________
#

#Sum the Column 'Weekly Sales' by Date
#Print all values from top_average_sales matrix
#TOP 10 DESCENDING (Sorted)
top_average_sales= unified_table.groupby(by=['Date'],as_index=False )['Weekly_Sales'].sum() 

print(top_average_sales.sort_values('Weekly_Sales',ascending= False).head(10)) 				
#No   Date         Weekly_Sales
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
#No   Date         Weekly_Sales
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


#***********************Line Chart***********************
plt.figure(figsize=(20,5))
plt.plot(average_sales_week.Date,average_sales_week.Weekly_Sales)	
plt.show()


#From unified_table we need to measure the mean values of fuel_price and temperatures.
fuel_price = unified_table.groupby(by=['Date'], as_index=False)['Fuel_Price'].mean()
temperature = unified_table.groupby(by=['Date'], as_index=False)['Temperature'].mean()



#***********************Combined Line Chart***********************
# 2 Y-AXIS GRAPH COMBINATION OF FUEL_PRICE AND TEMPERATURE (Y),DATE (X)
fig, ax1 = plt.subplots(figsize=(20,5)) 
ax1.plot(fuel_price.Date,fuel_price.Fuel_Price, 'g-' )
ax2 = ax1.twinx()
#SHOW US THE SEASONALITY
ax2.plot(temperature.Date,temperature.Temperature, 'b-')
plt.show() 



#***********************Combined Bar and Line Chart***********************
# 2 Y-AXIS GRAPH COMBINATION OF FUEL_PRICE AND AVERAGE SALES VOLUME (Y),DATE (X)
fig, ax1 = plt.subplots(figsize=(20,5)) 
ax1.plot(fuel_price.Date,fuel_price.Fuel_Price, 'b-' )
ax2 = ax1.twinx()
#SHOW US THE SEASONALITY
plt.bar(average_sales_week.Date,average_sales_week.Weekly_Sales, color = 'orange')
plt.ylabel("Avg Sales (10K)")
plt.show() 


#Top Store Sales Descending Sorted
top_sales = unified_table.groupby(by=['Store'],as_index=False)['Weekly_Sales'].sum()
print(top_sales.sort_values('Weekly_Sales',ascending=False)[:10])

# Store  Weekly_Sales
#  20	 3.013978e+08
#  4	 2.995440e+08
#  14 	 2.889999e+08
#  13 	 2.865177e+08


#________________________________________________________________________________________________________________________________________________

#
#_________________________________AUTOCORRELATION = SERIAL CORRELATION INTRODUCTION_________________________________
#


#Our problem fitting with Autocorrelation (Time Series Problem)
# Create the first Forecast Model for Total Sales Volume.


#IMPORT THE NECESSARY LIBRARY FOR THE AUTOCORRELATION AND PARTIAL AUTOCORRELATION
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf 
#acf = autocorrelation function
#pacf = partial autocorrelation function


#I make a small correlation analysis to see if one day has a strong or not relationship with the day before. 
#FROM average_sales_week DATAFRAME WE ARE USING FROM THE EXCISTING COLUMNS
#(DATE,AVG_SALES) WE ARE USING ONLY DATE
avg_sales = average_sales_week.set_index('Date') 

#From avg_sales we want only the weekly sales.
#We need the Weekly_Sales because in this column will run the correlation analysis.
x=avg_sales['Weekly_Sales']

#Set the variables with the days. day_i are all days, the day_i_minus has one less day, the last one.
day_i= x[1:]
day_i_minus = x[:-1]

#The number that measures how correlate are the day_i with the day_i_minus
np.corrcoef(day_i_minus,day_i)[0,1]
#From the result we understand that there is a correlation between the days, but the number doesn't give us the confidence that we want.
#0.3377879144700981

#________________________________________________________________________________________________________________________________________________

#_________________________________SCATTER PLOT WHO SHOW US THAT APPEARS A CORRELATION BETWEEN DAS_________________________________

#I present you in a scatter plot the distribution of avg sales of day i (x) and avg sales day i-1 (y).
#With this diagram we understand that i days and the day before i are highly correlated.
#We see that we don't have a WEAK Correlation between these days, and every day is dependent from the day before.
#It is a useful insight for our forecast model, because we are gonna based in unique weeks and we have to evaluate our results.
colors = np.random.rand(len(day_i))
area = (30 * np.random.rand(len(day_i)))**2  # 0 to 15 point radii
plt.scatter(day_i,day_i_minus,c = colors, alpha =0.6)
plt.xlabel("avg sales day i")
plt.ylabel("avg sales day i-1")



#________________________________________________________________________________________________________________________________________________

#
#_________________________________AUTOCORRELATION AND PARTIAL AUTOCORRELATION FOR AVG_SALES VOLUME_________________________________
#
#This is a pre analysis of avg_sales to how
#THIS IS THE AUTOCORRELATION PLOT WHICH INCLUDES X AXES = avg_sales, 
#We correlate a time series with its self.
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.core import datetools
fig, axes = plt.subplots(1,2, figsize=(22,8))
plot_acf(avg_sales, ax=axes[0])
plot_pacf(avg_sales, ax=axes[1])
plt.show()

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


#***************RESULTS************************
#We need to focus on the high value weeks. We see that
#1,52 weeks are positive correlated.
#6 week is negative correlated.


#________________________________________________________________________________________________________________________________________________

#
#_________________________________FORECAST MODEL DEFINITION_________________________________
#
#STEP BY STEP.
#https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

#Import Linear Regression 
#It is a manual implementation of the model, because I have to deal with consecutive seasonality terms.
from sklearn.linear_model import LinearRegression

#I made a new fuction called fit_ar_model which need the avg_sales and the weeks from the observation as input.
#The function return the coefficient and the intercept of those values.
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

    #Print the score factor is R^2
    print('Score factor: %.2f' % linear_regression.score(X[mask],Y[mask]))
    
    #Those are the variables that returns the model.
    return linear_regression.coef_, linear_regression.intercept_
    
#With the coefficient and the intercept that we was exported from the ar_model we give it as import to the prediction_model.
#The model uses all avg_sales and fills the array with values which satisfy the condition I have set.
#WE have 1-D arrays and .dot method create the inner product of vectors.
#All these values are summarized
def predict_ar_model(avg_sales, orders, coefficient, intercept):
    return np.array([np.sum(np.dot(coefficient, avg_sales.values[(i-weeks)].squeeze())) + intercept  if i >= np.max(weeks) else np.nan for i in range(len(avg_sales))])


#
#_________________________________Sales Prediction Version 1_________________________________
#
#Given the seasonality observed from the ACF and the PACF function, the AR model is implemented including seasonality from weeks (1,6,52).
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
#Score factor: 0.87 is R^2, It explains how well the linear model fits a set of observations. 


#After 52 week we will see the prediction line.
diff=(avg_sales['Weekly_Sales']-pred[0])/avg_sales['Weekly_Sales']
print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std()))
plt.figure(figsize=(20,5))
plt.plot(diff, c='orange')
plt.grid()
plt.show()
#AR Residuals: avg -0.00, std 0.03S


#
#_________________________________Sales Prediction Version 2_________________________________
#
#Given the seasonality observed from the ACF and the PACF function, the AR model is implemented including seasonality from weeks (1,6,52).

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


#After 57 week we will see the prediction line.
diff1=(avg_sales['Weekly_Sales']-pred[0])/avg_sales['Weekly_Sales']
print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std())) 
plt.figure(figsize=(20,5))
plt.plot(diff,'green')
plt.plot(diff1,'red')
plt.grid()
plt.show()
#AR Residuals: avg -0.00, std 0.04


#R^2 MEANING
#You should evaluate R-squared values in conjunction with residual plots, 
#other model statistics, and subject area knowledge in order to round out the picture (pardon the pun).
#0% indicates that the model explains none of the variability of the response data around its mean.
#100% indicates that the model explains all the variability of the response data around its mean.

#http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit


#________________________________________________________________________________________________________________________________________________

#
#_________________________________FORECAST MODEL DEFINITION FOR STORE 20_________________________________
#
#Focus on Store 20 which we saw before that makes the highest sales of all gas stations on the dataset.
#fs= focus_store

#I create a new table the fs to slice unified_table and bring to this array all the values of store 20.
fs=unified_table.where( unified_table['Store'] == 20)
#Removing missing values, because this will effect negativly the analysis.
fs=fs.dropna()
#For the analysis we want the sales, so I groupby date all the Weekly_Sales
fs=fs.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
fs = fs.set_index('Date')
df = pd.DataFrame(fs)
#I make a transformation in array and I create a new Column with name 'week' to change from date to number of weeks.
#There are no calendar weeks but a row numbering.
df['week']=df.reset_index().index
fs

#I present the sales values by week in a Line Chart.
plt.figure(figsize=(20,5))
plt.plot(fs.week, fs.values)
plt.show()

#I run again the Autocorrelation and Partial Correlation code,this time with fsw array.
#THE 2-D fs, make it 1-D fsw
fsw = fs.set_index('week')
fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(fsw.values, lags=100, alpha=0.05, ax=axes[0])
plot_pacf(fsw.values, lags=100, alpha=0.05, ax=axes[1])
plt.show()


#
#_________________________________Sales Prediction Version 1 for store 20_________________________________
#
#I set the consecutive seasonality terms. The values 1,6,29 was picked from the autocorrelation analysis (highest values positive or negative )
weeks=np.array([1,2,3,5,6,7,10,29,38,39,40,41,43,47,48,51,52])
#Call the function and set the returning values.
coef, intercept = fit_ar_model(fsw,weeks)
#Call the prediction function
pred=pd.DataFrame(index=fsw.index, data=predict_ar_model(fsw, weeks, coef, intercept))
#Presenting the resulting from the fsw array (original values) combined with the predicted values of the algorithm.
#The predictions are after 52 week.
plt.figure(figsize=(20,5))
plt.plot(fsw, 'b')
plt.plot(pred, 'r')
plt.show()
#[[ 0.12007631 -0.02882628 -0.00895904  0.03520367  0.75093771]] [309689.73529126]
#Score factor: 0.75, It explains how well the linear model fits a set of observations. 


#This graph shows us the differences from the original values and the redicted during the weeks. After 52 week.
diff2=(fsw['Weekly_Sales']-pred[0])/fsw['Weekly_Sales']
print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std()))
plt.figure(figsize=(20,5))
plt.plot(diff2,c='orange')
plt.grid()
plt.show()
#
#_________________________________Sales Prediction Version 2 for store 20_________________________________
#
#I set the consecutive seasonality terms. The values 1,6,29 was picked from the autocorrelation analysis (highest values positive or negative )
weeks=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52])
#Call the function and set the returning values.
coef, intercept = fit_ar_model(fsw,weeks)
#Call the prediction function
pred=pd.DataFrame(index=fsw.index, data=predict_ar_model(fsw, weeks, coef, intercept))
#Presenting the resulting from the fsw array (original values) combined with the predicted values of the algorithm.
#The predictions are after 52 week.
plt.figure(figsize=(20,5))
plt.plot(fsw, 'b')
plt.plot(pred, 'r')
plt.show()
#[[ 0.14423516 -0.06246291 -0.02116239 -0.00926447  0.00350683  0.03973046 0.75616876]] [347087.57553849]
#Score factor: 0.76, It explains how well the linear model fits a set of observations. 

#This graph shows us the differences from the original values and the redicted during the weeks. After 52 week.diff=(fsw['Weekly_Sales']-pred[0])/fsw['Weekly_Sales']
diff3=(fsw['Weekly_Sales']-pred[0])/fsw['Weekly_Sales']
print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std()))
plt.figure(figsize=(20,5))
plt.plot(diff1,'green')
plt.plot(diff3,'red')
plt.grid()
plt.show()
#AR Residuals: avg -0.00, std 0.05


#________________________________________________________________________________________________________________________________________________


#
#_________________________________EXTRA ANALYSIS ON REMAINING DATA_________________________________
#

#From the unified_table we need to slice all the columns that we are going to analyzed.
extra_analysis=unified_table.where( unified_table['Store'] == 20)
extra_analysis=extra_analysis.dropna()
extra_analysis=extra_analysis.groupby(by=['Date'], as_index=False)[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                                                  'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].mean()
extra_analysis = extra_analysis.set_index('Date')
extra_analysis.head()


#We need to take a quick look at the variables.This function compute mean,std,min,max, etc of each column. 
extra_analysis.describe()


#One technique that we used before was the shifting. We need to shift the days (-1) and reran the whole analysis,
# to see if the correlation between the 
#variables are the same and if there are some insights. I took the fsw. 
#I create a new column in extra_analysis array the 'shifted_sales' and in this 
#column I will run the correlation analysis.
extra_analysis['shifted_sales'] = fs.shift(-1)
extra_analysis.head()


#Correlation analysis for all the values in extra_analysis table.
import seaborn as sns
corrmat = extra_analysis.corr()
f, ax = plt.subplots(figsize=(12,9)) 
sns.heatmap(corrmat,vmax=1, square=True, annot=True ); 


corr['shifted_sales'].sort_values(ascending=False)
#________________________________________________________________________________________________________________________________________________





#
#_________________________________Re-write the algorithm using these info from the extra_analysis_________________________________
#
def fit_ar_model_ext(avg_sales, weeks, ext):
    
    X=np.array([ avg_sales.values[(i-weeks)].squeeze() if i >= np.max(weeks) else np.array(len(weeks) * [np.nan]) for i in range(len(avg_sales))])
    
    X = np.append(X, ext.values, axis=1)
    
    mask = ~np.isnan(X[:,:1]).squeeze()
    
    Y= avg_sales.values
    
    linear_regression=LinearRegression()
    
    linear_regression.fit(X[mask],Y[mask].ravel())
    
    print(linear_regression.coef_, linear_regression.intercept_)

    print('Score factor: %.2f' % linear_regression.score(X[mask],Y[mask]))
    
    return linear_regression.coef_, linear_regression.intercept_
    
def predict_ar_model_ext(avg_sales, weeks, ext, coef, intercept):

    X=np.array([ avg_sales.values[(i-weeks)].squeeze() if i >= np.max(weeks) else np.array(len(weeks) * [np.nan]) for i in range(len(avg_sales))])
    
    X = np.append(X, ext.values, axis=1)
    
    return np.array( np.dot(X, coef.T) + intercept)
#________________________________________________________________________________________________________________________________________________


#
#___________________In shifted_sales I have one value(the last one which is NaN) I have to replace it_________________________________
#

info = pd.DataFrame(extra_analysis.dtypes).T.rename(index = {0:'Column Type'})	
info = info.append(pd.DataFrame(extra_analysis.isnull().sum()).T.rename(index = {0:'null values (nb)'}))
info
#Fill all the NaN values with 0.
extra_analysis.fillna(0, inplace=True)
   

weeks=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52])
coef, intercept = fit_ar_model_ext(fsw,weeks,extra_analysis)
pred_ext=pd.DataFrame(index=fsw.index, data=predict_ar_model_ext(fsw, weeks, extra_analysis, coef, intercept))
plt.figure(figsize=(20,5))
plt.plot(fsw, 'o')
plt.plot(pred)
plt.plot(pred_ext)
plt.show()


plt.figure(figsize=(20,5))
plt.plot(fsw, 'orange')
plt.plot(pred_ext,'r')
plt.show()


diff4=(fsw['Weekly_Sales']-pred[0])/fsw['Weekly_Sales']
diff_ext4=(fsw['Weekly_Sales']-pred_ext[0])/fsw['Weekly_Sales']

print('AR Residuals: avg %.2f, std %.2f' % (diff4.mean(), diff4.std()))
print('AR wiht Ext Residuals: avg %.2f, std %.2f' % (diff_ext4.mean(), diff_ext4.std()))
 
plt.figure(figsize=(20,5))
plt.plot(diff4, c='green', label='w/o external variables')
plt.plot(diff_ext4, c='red', label='w/ external variables')
plt.legend()
plt.grid()
plt.show()
#________________________________________________________________________________________________________________________________________________

#
#_________________________________REGRESSION ANALYSIS_________________________________
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
