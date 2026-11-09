import Class_Task
task = Class_Task.Task
import Class_Task_Manager
manager = Class_Task_Manager.TaskManager()
while True:
    print("1.Add")
    print("2.Show")
    print("3.Save")
    print("4.Load")
    print("5.Remove")
    print("6.Edit")
    print("7.The Most important Task")
    print("8.Sort by Priority")
    print("9.Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        mozoo = input("Title: ")
        ahamiat = input("Priority: ")
        vazife = task(mozoo, ahamiat)
        manager.add_tasks(vazife)
    elif choice == "2":
        manager.show_info()
    elif choice == "3":
        manager.save_tasks()
    elif choice == "4":
        manager.load_tasks()
    elif choice == "5":
        title = str(input("Title: "))
        manager.remove_tasks(title)
    elif choice == "6":
        title = str(input("Title: "))
        manager.edit_tasks(title)
    elif choice == "7":
        manager.show_top_task()
    elif choice == "8":
        manager.sort_by_priority()
    elif choice == "9":
        break
