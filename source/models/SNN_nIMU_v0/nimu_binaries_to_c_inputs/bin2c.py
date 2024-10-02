

foo         = "west"
inf     = open("inputs_outputs/" + foo + "/in.txt", 'r')
outf    = open(foo + ".txt", 'w')
contents = inf.readlines() #put the lines to a variable (list).
string = '{'
for l in contents:
    string += '{'
    for letter in l:
        string += letter
        string += ','
    string = string[:-3]
    string += "},"
string = string[:-1]
string += '}'
outf.write(string)
inf.close()
outf.close()

