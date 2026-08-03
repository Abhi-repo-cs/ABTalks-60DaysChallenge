'''take user input'''

name=input(" Enter Your Name: ")
age=int(input(" Enter Your age: "))
salary=float(input(" Enter your monthly salary in Numerical Format: "))

'''Salary Calculation'''

yearly_salary=salary*12

dict={
    "name":name,
    "age":age,
    "Annual salary": yearly_salary
}

print(dict)