import re
# ----------------------------------------------------------------------------------------------------------------------
''' re.sub()
    re.find()
    re.match()
    re.search()
    re.findall() '''
# ----------------------------------------------------------------------------------------------------------------------
sentence1 = "My name is Amin and I was born in 1995 and I am learning AI in 2026"
print(re.findall(r'\d', sentence1))            # digits
print(re.findall(r'\d*', sentence1))           # 0 or more digits
print(re.findall(r'\d+', sentence1))           # 1 or more digits
print(re.findall(r'\w', sentence1))            # alphanumeric characters
print(re.findall(r'\w+', sentence1))           # 1 or more characters
print(re.findall(r'[ami]', sentence1))         # Anything in []
print(re.findall(r'[ami]*', sentence1))
print(re.findall(r'[ami]+', sentence1))
print(re.findall(r'[a-z]+', sentence1))
print(re.findall(r'[a-zA-Z]+', sentence1))
print(re.findall(r'[a-zA-Z0-9]+', sentence1))
# ----------------------------------------------------------------------------------------------------------------------
sentence2 = "My name is Amin. Today is 2026-08-11. You can contact me at amin_ai.work@gmail.com, (012) 345-6789 after 2027/01/01."
print(re.findall(r'\d{4}-\d{2}-\d{2}', sentence2))      # 1st Date format
print(re.findall(r'\d{4}/\d{2}/\d{2}', sentence2))      # 2nd Date format
print(re.findall(r'[\w_\.]+@\w+.com', sentence2))       # E-Mail format
print(re.findall(r'\(\d{3}\) \d{3}-\d{4}', sentence2))  # Phone-Number format
