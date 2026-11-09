import pandas as pd
df = pd.read_csv('pokemon_data.csv')
df_xlsx = pd.read_excel('pokemon_data.xlsx')
df_txt = pd.read_csv('pokemon_data.txt', delimiter = '\t')
#------------------------------------------------------------------------------------
print ("Importing The Entire CSV:")
print (df)
print ("-----------------------------------------------------------------------------")
print ("Importing Excel Head:")
print (df_xlsx.head())
print("-----------------------------------------------------------------------------")
print("Importing Text_file Tail:")
print (df_txt.tail(3))
