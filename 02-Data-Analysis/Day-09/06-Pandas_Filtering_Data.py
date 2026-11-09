import pandas as pd
df = pd.read_csv('pokemon_data.csv')
#------------------------------------------------------------------------------------
print ("Filtering by a column's specific value:")
print (df.loc[df['Type 1'] == 'Grass'])
print ("-----------------------------------------------------------------------------")
print ("Filtering by multiple columns' specific values (OR):")
print (df.loc[(df['Type 1'] == 'Grass') | (df['Type 2'] == 'Poison')])
print ("-----------------------------------------------------------------------------")
print ("Filtering by multiple columns' specific values (AND):")
print (df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') & (df['HP'] > 70)])
print ("-----------------------------------------------------------------------------")
print ("Resetting index in filtered data:")
new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') & (df['HP'] > 70)]
print (new_df.reset_index(drop=True, inplace=True))
print ("-----------------------------------------------------------------------------")
print ("Filtering the rows their \"Name\" contains \"Mega\":")
print (df.loc[df['Name'].str.contains('Mega')])
print ("-----------------------------------------------------------------------------")
print ("Filtering the rows their \"Name\" doesn't contain \"Mega\":")
print (df.loc[~df['Name'].str.contains('Mega')])
print ("-----------------------------------------------------------------------------")
print ("Filtering the rows their \"Type 1\" contains \"Grass\" or \"Fire\":")
import re
print (df.loc[df['Type 1'].str.contains('Grass|Fire', regex=True)])
# In this command, CAPITALIZATION is important
# To ignore it you can write the command like:
# df.loc[df['Type 1'].str.contains('grass|fire', flags=re.I, regex=True)]
print ("-----------------------------------------------------------------------------")
print ("Filtering the rows their \"Name\" starts with \"Pi\" or \"pi\":")
print (df.loc[df['Name'].str.contains('^pi', flags=re.I, regex=True)])
print ("-----------------------------------------------------------------------------")
