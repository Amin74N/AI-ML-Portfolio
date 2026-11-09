class Product:
    def __init__(self,name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
    def show_info(self):
        print(self.name, self.price, self.stock)
    def buy(self, amount):
        self.stock -= amount
    def add_stock(self, amount):
        self.stock += amount
product1 = Product("Phone", "100000000", 74)
product1.show_info()
product1.buy(5)
product1.show_info()
product1.add_stock(4)
product1.show_info()