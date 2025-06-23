# 作者: 王梓豪
# 2025年06月02日22时15分15秒
# 2958126254@qq.com

class Dog:
    def __init__(self,name,color):
        self.name = name
        self.color = color
    def run(self):
        print('run')
    def play(self):
        print("摇尾巴")

xiaohong = Dog('xiaohong','yellow')
xiaohong.play()
xiaohong.run()