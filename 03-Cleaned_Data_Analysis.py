import pandas as pd
#--------------------------------------------------------------------
df = pd.read_csv("clean_sales.csv")
#--------------------------------------------------------------------
print("Number of products sold:", df["quantity"].sum())
print('--------------------------------------------------------------------')
print("Average product price:", df["price"].mean())
print('--------------------------------------------------------------------')
print("The most expensive product:\n", df[df["price"] == df["price"].max()][["product", "price"]].iloc[0])
print('--------------------------------------------------------------------')
print("The best-selling product:\n", df.groupby("product")["quantity"].sum().sort_values(ascending= False).head(1))
print('--------------------------------------------------------------------')
print("Average customer rating:", df["rating"].mean())
print('--------------------------------------------------------------------')
print("Sales by city:\n", df.groupby("city")["price"].sum().sort_values(ascending= False))
print('--------------------------------------------------------------------')
print("Sales by product:\n", df.groupby("product")["quantity"].sum().sort_values(ascending= False))
print('--------------------------------------------------------------------')
print("The most popular product:\n", df.groupby("product")["rating"].max().sort_values(ascending= False).head(1))
