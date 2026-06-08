import json
import Class_Task
task = Class_Task.Task
class TaskManager():
    def __init__(self):
        self.task_list = []
    def add_tasks(self, task):
        self.task_list.append(task)
    def remove_tasks(self, title):
        for task in self.task_list[:]:
            if task.title == title:
                self.task_list.remove(task)
                return
        print(title + " not found!")
    def show_info(self):
        if not self.task_list:
            print("Empty List!")
            return
        for task in self.task_list:
            print(task.priority, task.title)
    def save_tasks(self, filename="Tasks.json"):
        try:
            data = []
            for task in self.task_list:
                data.append(task.json_write())
                py_file = open(filename, "w")
                json.dump(data, py_file)
        except:
            print("Empty List!")
    def load_tasks(self, filename="Tasks.json"):
         try:
             py_file = open(filename, "r")
             data = json.load(py_file)
             self.task_list = []
             for item in data:
                 vazife = task.json_read(item)
                 self.task_list.append(vazife)
         except FileNotFoundError:
             print("Empty List!")
    def edit_tasks(self, title):
        for task in self.task_list[:]:
            if task.title == title:
                choice = str(input("1.Change Title\n2.Change Priority\n"))
                if choice == "1":
                    task.title = str(input("New Title: "))
                elif choice == "2":
                    task.priority = int(input("New Priority: "))
                else:
                    print("Invalid Choice!")
                return
        print(title + " not found!")
    def show_top_task(self):
        if not self.task_list:
            print("Empty List!")
            return
        max_priority = 1000
        top_task = []
        for task in self.task_list:
            if max_priority >= task.priority:
                max_priority = task.priority
                top_task = task
        print("The most important task is: " + top_task.title)
    def sort_by_priority(self):
        if not self.task_list:
            print("Empty List!")
            return
        self.task_list.sort(key=lambda t: t.priority)
