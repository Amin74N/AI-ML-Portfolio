'''
There are 4 different modes you can read files:
r (read): you only can read what is in the file without modification
w (write): you can write new information as well as change existing information
a (append): you can append information onto the end of the file without modification or change
r+ (read & write): you will have all the power of reading and writing
'''
employee_file = open("Employees-R.txt", "r")
print (employee_file.readable())
print (employee_file.read())
employee_file.close()
employee_file = open("Employees-R.txt", "r")
print (employee_file.readline())
employee_file.close()
employee_file = open("Employees-R.txt", "r")
print (employee_file.readlines())
employee_file.close()
employee_file = open("Employees-R.txt", "r")
print (employee_file.readlines()[1])
employee_file.close()
employee_file = open("Employees-R.txt", "r")
for employee in employee_file.readlines():
    print (employee)
employee_file.close()
