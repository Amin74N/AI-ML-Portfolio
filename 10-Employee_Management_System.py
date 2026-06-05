class Employee:
    def __init__(self,name, salary, department):
        self.name = name
        self.salary = salary
        self.department = department
    def show_info(self):
        print(self.name, self.salary, self.department)
    def increase_salary(self, amount):
        self.salary += amount
employee1 = Employee("Amin", 50000000, "IT/AI")
employee1.show_info()
employee1.increase_salary(5000000)
employee1.show_info()
