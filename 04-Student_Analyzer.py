import csv
csv_file = open("Student.csv", "a", newline='')
csv_writer = csv.writer(csv_file)
while True:
        choice = input ("New student? Y/N").lower()
        if choice == "y":
            name = input("Enter Student Name: ")
            score = float(input("Enter Score: "))
            csv_writer.writerow ([name, score])
        else:
            break
csv_file.close()
csv_file = open("Student.csv", "r")
csv_reader = csv.reader(csv_file)
total = 0
i = 0
top_score = 0.0
student = ""
next(csv_reader)
for row in csv_reader:
    total += float(row[1])
    if float(row[1]) >= top_score:
        top_score = float(row[1])
        student = row[0]
    i += 1
csv_file.close()
import json
JSON_Report = {
    "Student Count": str(i),
    "Average Score": str(total/i),
    "Top Student": student,
    "Top Score": str(top_score)
}
Python_File = open("JSON_Report", "w")
json.dump(JSON_Report, Python_File)
Python_File.close()
