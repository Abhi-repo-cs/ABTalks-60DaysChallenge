import csv
grade_count={}
with open("students.csv") as f:
    print(f.readline()) #to check the file has been loaded
    reader =csv.reader(f)
    next(reader) #to skip the title

    for row in reader:
        grade=row[2]

        if grade in grade_count:
            grade_count[grade]+=1
        else:
            grade_count[grade]=1

    print(grade_count)

with open("output.txt","x") as file:
    file.write(str(grade_count))
    print("the file has been written successfully")
    
