import json
class Task:
    def __init__(self, mozoo, ahamiat):
        self.title = mozoo
        self.priority = int(ahamiat)
    def json_write(self):
        return {
            "title": self.title,
            "priority": self.priority,
        }
    def json_read(data):
        return Task(data["title"], data["priority"])
