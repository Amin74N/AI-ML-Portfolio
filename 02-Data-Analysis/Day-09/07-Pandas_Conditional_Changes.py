import pandas as pd
df = pd.read_csv('pokemon_data.csv')
#------------------------------------------------------------------------------------
print ("Changing the rows their \"Type 1\" Value is \"Fire\" to \"Flamer\":")
df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flamer'
print (df)
print ("-----------------------------------------------------------------------------")
print ("Changing the \"Type 1 = Flamer\" rows' \"Legendary\" Value to \"True\":")
df.loc[df['Type 1'] == 'Flamer', 'Legendary'] = True
print (df)
print ("-----------------------------------------------------------------------------")
print ("Changing the \"HP > 80\" rows' \"Generation\" & \"Legendary\" Values to \"100\" & \"False\":")
df.loc[df['HP'] > 80, ['Generation','Legendary']] = [100,False]
print (df)
print ("-----------------------------------------------------------------------------")
