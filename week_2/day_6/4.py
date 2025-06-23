# 作者: 王梓豪
# 2025年06月02日22时17分21秒
# 2958126254@qq.com

class Gun:
    def __init__(self, module):
        self.module = module
        self.bullet_count = 0
    def add_bullet(self,count):
        self.bullet_count += count
    def shoot(self):
        if(self.bullet_count <= 0):
            print('请装弹')
            return
        else:
            self.bullet_count -= 1
            print("%s 发射子弹[%d]..." % (self.module, self.bullet_count))

class Soldier:
    def __init__(self,name,age,height,weight):
        self.name = name
        self.age = age
        self.height = height
        self.weight = weight
        self.gun = None
    def fire(self):
        if not self.gun:
            print('士兵没有枪')
            return
        else:
            self.gun.add_bullet(50)
            self.gun.shoot()

gun=Gun('AK47')
xusanduo=Soldier('许三多',18,180,75)
xusanduo.gun=gun
xusanduo.fire()

a=xusanduo
print(a is xusanduo)
