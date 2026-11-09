import pandas as pd
df = pd.read_csv('pokemon_data.csv')
#------------------------------------------------------------------------------------
#Saving Data in a CSV File:
df.to_csv('modified_data.csv')
#File "modified_data.csv" is saved in your directory
#------------------------------------------------------------------------------------
#Saving Data in a CSV File Without Index:
df.to_csv('modified_data.csv', index=False)
#File "modified_data.csv" is saved in your directory
#------------------------------------------------------------------------------------
#Saving data in an Excel File:
df.to_excel('modified_data.xlsx', index=False)
#File "modified_data.xlsx" is saved in your directory
#------------------------------------------------------------------------------------
#Saving data in a Text File:
df.to_csv('modified_data.txt', index=False, sep='\t')
#File "modified_data.txt" is saved in your directory
