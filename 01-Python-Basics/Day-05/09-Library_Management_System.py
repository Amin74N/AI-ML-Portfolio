class Book:
    def __init__(self,title, author, available):
        self.title = title
        self.author = author
        self.available = available
    def show_info(self):
        print("Title:", self.title, "\nAuthor:", self.author, "\nAvailability:", self.available)
    def borrow(self, amount):
        print("This book has been borrowed", amount, "times")
    def return_book(self, amount):
        print("This book has returned back", amount, "times")
book1 = Book("Python", "Amin", False)
book1.show_info()
book1.borrow(5)
book1.return_book(4)