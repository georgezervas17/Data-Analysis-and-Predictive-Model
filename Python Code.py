In[1]
import numpy as np #I use numpy for Linear Algebra
import pandas as pd #I use pandas for data processing, I have to upload my CSV Files

In[2]
features = pd.read_csv("CSV Features.csv") #Load the 3 CSV that we are gonna
sales = pd.read_csv("CSV Sales.csv")
stores = pd.read_csv("CSV Stores.csv")
																#In Anaconda prompt you have to install the PrettyTable
In[3]															#Command: 'easy_install prettytable'
data_inspection = PrettyTable(['Table Name','Table Dimension']) #Use PrettyTable to print the shape of CSV
data_inspection.add_row(['features',store.shape])
data_inspection.add_row(['sales',sales.shape])
data_inspection.add_row(['stores',stores.shape])
print(data_inspection,'\n')

In[4]
print('Features: ', features.columns.tolist()) #Print the Column Names of each CSV
print('Sales: ',sales.columns.tolist())
print('Stores: ', stores.columns.tolist())

In[5]
unified_table = sales.merge(features,how="left", on=['Store', 'Date', 'IsHoliday']) #Union all the columns in one 
unified_table = unified_table.merge(stores,how ="left",on=['Store'])				#table so that can easy to correlate
unified_table.head()															    #all the values, I use pandas DateFrame

 *******************Problem************************
#unified_table = features.merge(sales,how="left", on=['Store', 'Date', 'IsHoliday']) #Union all the columns in one 
#unified_table = unified_table.merge(stores,how ="left",on=['Store'])				#table so that can easy to correlate
#unified_table.head()	


In[6]
import matplotlib.pyplot as plt #import matplotlib for the correlation

In[7]
corrmat = unified_table[['Store','Date','Weekly_Sales','Temperature','Fuel_Price','CPI',
						 'Unemployment','Type','IsHoliday','MarkDown1',
						 'MarkDown2','MarkDown3','MarkDown4','MarkDown5',
						 'Size','Dept']].corr() #I use the unified table to correlate the values

In[8]
import seaborn as sns #I need this import for the results of the correlation

In[9]
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=1, square=True, annot=True ); #We have to change the values in
													   #because it doesn't correlate those values 
													   #(Denormalization)
													   #Temperature,Fuel_Price,CPI,Unenployment

In[10]
info = pd.DataFrame(unified_table.dtypes).T.rename(index = {0:'Column Type'}) #Identify how many values are NaN from each column
info = info.append(pd.DataFrame(unified_table.isnull().sum()).T.rename(index = {0:'null values (nb)'}))
info

In[11]
unified_table.fillna(0, inplace=True)