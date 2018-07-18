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
final = sales.merge(features,how="left", on=['Store','Date','IsHoliday'] ) #Union all the columns in one 
final = final.merge(stores,how ="left", on=['Store'])					   #table so that can easy to correlate
final.head()															   #all the values