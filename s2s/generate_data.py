import numpy as np
dictionary = "abcdefghijklmnopqrstuvwxyz"
source = []
for i in range(1000):    
    string_length = int(np.random.uniform(2,10))
    string = ""
    for j in range(string_length):
        random_char = int(np.random.uniform(0,26))
        string = string + dictionary[random_char]
    string = string+"\n"
    source.append(string)

with open("source",'w') as f:
    f.writelines(source)





with open("source") as f:
    lines = f.readlines()

target = []
for line in lines:
    line = line[:-1]
    target.append(line[::-1])
    target[-1]=target[-1]+"\n"

print(target)

with open("target",'w') as f:
    f.writelines(target)

