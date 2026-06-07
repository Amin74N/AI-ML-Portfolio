import json
import Class_Student
Student = Class_Student.Student
class StudentManager():
    def __init__(self):
        self.student_list = []
    def add_student(self, student):
        self.student_list.append(student)
    def remove_student(self, name):
        for student in self.student_list[:]:
            if student.name == name:
                self.student_list.remove(student)
                return
        print(name + " not found!")
    def show_info(self):
        if not self.student_list:
            print("Empty List!")
            return
        for student in self.student_list:
            print(student.name, student.age, student.score)
    def save_students(self, filename="Students.json"):
        try:
            data = []
            for student in self.student_list:
                data.append(student.json_write())
                py_file = open(filename, "w")
                json.dump(data, py_file)
        except:
            print("Empty List!")
    def load_students(self, filename="Students.json"):
         try:
             py_file = open(filename, "r")
             data = json.load(py_file)
             self.student_list = []
             for item in data:
                 student = Student.json_read(item)
                 self.student_list.append(student)
         except FileNotFoundError:
             print("Empty List!")
    def show_average(self):
        if not self.student_list:
            print("Empty List!")
            return
        sum = 0
        for student in self.student_list:
            sum += student.score
        avg = sum / len(self.student_list)
        print("Average Score: " + str(avg))
    def show_top_student(self):
        if not self.student_list:
            print("Empty List!")
            return
        max_score = 0
        top_student = []
        for student in self.student_list:
            if max_score <= student.score:
                max_score = student.score
                top_student = student
        print("Top Student: " + top_student.name + " With Score: " + str(max_score))
    def sort_by_score(self):
        if not self.student_list:
            print("Empty List!")
            return
        self.student_list.sort(key=lambda s: s.score)
