# 作者: 王梓豪
# 2025年06月02日23时17分13秒
# 2958126254@qq.com

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def eat(self):
        print("吃---")

    def drink(self):
        print("喝---")

    def run(self):
        print("跑---")

    def sleep(self):
        print("睡---")
    def TY_test(self):
        pass

class Women(Person):
    def __init__(self, name, age):
        self.name = name
        self.__age = age

    def __get_age(self):
        print(self.__age)

    def boy_friend(self):
        self.__get_age()
    def TY_test(self):
        print('800 m')

class Men(Person):
    def __init__(self, name, age):
        super().__init__(name,age)
    def TY_test(self):
        print('1000 m')

xiaohong = Women('xiaohong',18)
xiaoming = Men('xiaoming',18)

xiaohong.boy_friend()
print(xiaoming.age)
xiaohong.TY_test()
xiaoming.TY_test()