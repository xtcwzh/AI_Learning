# 作者: 王梓豪
# 2025年06月06日22时15分33秒
# 2958126254@qq.com
from operator import itemgetter, attrgetter

list1 = [1, 3, 2, 4, 5, 3, 2]
print(sorted(list1))
print(list1)
list1.sort()
print(list1)
print('-'*50)

dict1 = {1: 'D', 3: 'B', 2: 'B', 4: 'E', 5: 'A'}
print(sorted(dict1))
print('-'*50)

str_list = "This is a test string from Andrew".split()
print(str_list)
print(sorted(str_list))
print('-'*50)

def cmp(str1:str):
    return str1.lower()

print(sorted(str_list , key = cmp))
print(sorted(str_list ,key = lambda str1:str1.lower()))
print('-'*50)


student_tuples = [
    ('john', 'A', 15),
    ('jane', 'B', 12),
    ('dave', 'B', 10),
]

def cmp_1(student):
    return student[1]

print(sorted(student_tuples,key = cmp_1))
print(sorted(student_tuples,key = lambda x:x[1]))
print(sorted(student_tuples,key = itemgetter(1)))
print('-' * 50)

class Student:
    def __init__(self,name,grade,age):
        self.name =name
        self.grade = grade
        self.age = age

    def __repr__(self):
        return repr((self.name,self.grade,self.age))

student_objects = [
    Student('john', 'A', 15),
    Student('jane', 'B', 12),
    Student('dave', 'B', 10),
]

print(sorted(student_objects,key = lambda x:x.age))
print(sorted(student_objects,key = attrgetter('age'),reverse = True))
print(sorted(student_objects,key = attrgetter('grade','age')))
print(sorted(student_objects,key = lambda x:(x.grade,x.age)))
print('-' * 50)

data = [('red', 1), ('blue', 1), ('red', 2), ('blue', 2)]
print(sorted(data,key = itemgetter(0)))
print('-' * 50)

mydict = { 'Li'   : ['M',7],
           'Zhang': ['E',2],
           'Wang' : ['P',3],
           'Du'   : ['C',2],
           'Ma'   : ['C',9],
           'Zhe'  : ['H',7] }
print(sorted(mydict.items(),key = lambda x : x[1][1]))
print('-' * 50)

gameresult = [
    { "name":"Bob", "wins":10, "losses":3, "rating":75.00 },
    { "name":"David", "wins":3, "losses":5, "rating":57.00 },
    { "name":"Carol", "wins":4, "losses":5, "rating":57.00 },
    { "name":"Patty", "wins":9, "losses":3, "rating": 71.48 }]
print(sorted(gameresult,key = lambda x : x['rating']))

tuples1=[(3,5),(1,2),(2,4),(3,1),(1,3)]
print(sorted(tuples1,key = lambda x: (x[0],-x[1])))
