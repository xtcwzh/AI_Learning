# 作者: 王梓豪
# 2025年06月02日18时49分42秒
# 2958126254@qq.com
from pyasn1_modules.rfc2985 import gender


def demo(num,*args,**kwargs):
    print(num)
    print(args)
    print(kwargs)

demo(1,2,3,4,5,6,7,8,9,name = 'xiaoming',age = 18,gender = 'true')
print('-' * 50 )

def demo2(num = 10):
    print(num)
demo2(name = 'xiaoming')
