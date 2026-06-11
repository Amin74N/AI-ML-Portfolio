import pandas as pd
df = pd.read_csv('pokemon_data.csv')
#------------------------------------------------------------------------------------
print ("Showing CSV File:")
print (df)
print ("-----------------------------------------------------------------------------")
print ("Adding A Column:")
df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
#It can also be done like this:
# df['Total'] = df.iloc[:,4:10].sum(axis=1)
print (df)
print ("-----------------------------------------------------------------------------")
print ("Changing The Added Column's Position:")
cols = list(df.columns)
df = df[cols[0:1]+[cols[-1]]+cols[1:12]]
print (df)
print ("-----------------------------------------------------------------------------")
print ("Dropping The Added Column:")
df = df.drop(columns=['Total'])
print (df)
