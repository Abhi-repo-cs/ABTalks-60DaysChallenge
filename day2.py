''' takes student marks'''

n=int(input("Enter the number of subjects: "))
total=0;

for i in range(n):
    
    marks=int(input(f"Enter marks for Subject {i+1}"))
    if(marks>=0 and marks<=100):
        total+=marks
    else:
        print("Enter valid marks!")

avg=(total/n)

print("The average marks =" ,avg)

if(avg>=90 and avg<100):
    print("Excellant")

elif(avg>=80 and avg<90):
    print("Very Good!")

elif(avg>=70 and avg<80):
    print("Good")

elif(avg>=60 and avg<70):
    print("Great Efforts!")

elif(avg>=50 and avg<60):
    print("good can do better")

elif(avg>=40 and avg<50):
    print("good needs improvement")

else:
    print("aim higher! better luck next time")