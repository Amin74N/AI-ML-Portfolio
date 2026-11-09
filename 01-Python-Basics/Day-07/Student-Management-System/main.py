import Class_Student
Student = Class_Student.Student
import Class_Student_Manager
manager = Class_Student_Manager.StudentManager()
while True:
    print("1.Add")
    print("2.Show")
    print("3.Save")
    print("4.Load")
    print("5.Remove")
    print("6.Average")
    print("7.Top Student")
    print("8.Sort by Score")
    print("9.Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        esm = input("Name: ")
        sen = input("Age: ")
        nomreh = input("Score: ")
        person = Student(esm, sen, nomreh)
        manager.add_student(person)
    elif choice == "2":
        manager.show_info()
    elif choice == "3":
        manager.save_students()
    elif choice == "4":
        manager.load_students()
    elif choice == "5":
        name = str(input("Name: "))
        manager.remove_student(name)
    elif choice == "6":
        manager.show_average()
    elif choice == "7":
        manager.show_top_student()
    elif choice == "8":
        manager.sort_by_score()
    elif choice == "9":
        break
