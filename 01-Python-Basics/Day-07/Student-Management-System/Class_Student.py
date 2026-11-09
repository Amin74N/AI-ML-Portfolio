import json
class Student:
    def __init__(self, esm, sen, nomreh):
        self.name = esm
        self.age = int(sen)
        self.score = float(nomreh)
    def json_write(self):
        return {
            "name": self.name,
            "age": self.age,
            "score": self.score
        }
    def json_read(data):
        return Student(data["name"], data["age"], data["score"])
