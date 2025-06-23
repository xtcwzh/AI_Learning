# 作者: 王梓豪
# 2025年06月09日20时44分34秒
# 2958126254@qq.com

import re
from re import match
from tkinter.font import names

result = re.match('wangdao','wangdao1.cn')
print(result.group())

ret = re.match('.','M,m')
print(ret.group())

ret = re.match('t.o','too')
print(ret.group())

ret = re.match('t.o','two')
print(ret.group())

ret = re.match('h','hello python')
print(ret.group())

res = re.match('[Hh]ello','hello')
print(res.group())

ret = re.match('[Hh]ello','Hello')
print(ret.group())

ret = re.match('[0,1,2,3,4,5,6,7,8,9]hello','5hello')
print(ret.group())

ret = re.match('[0-9]hello','5hello')
print(ret.group())

# ret = re.match('[0-35-9]hello','4hello')
# print(ret.group())

ret = re.match('[0-35-9]hello','5hello')#匹配0-3，5-9中的任意一个数字
print(ret.group())

ret = re.match('嫦娥\\d号','嫦娥1号发射')#什么都不加可能会被当成转义字符
print(ret.group())

ret = re.match(r'嫦娥\d号','嫦娥2号发射')
print(ret.group())

ret = re.match('[A-Z][a-z]*','MasdkhjsM')
print(ret.group())

names = ["name1","_name","2_name","__name__"]
for name in names:
    ret = re.match(r"[a-zA-Z_]+\w*",name)
    if ret:
        print(f"{name}符合要求")
    else:
        print(f"{name}不符合要求")

ret = re.match("[1-9]?[0-9]","23")
print(ret.group())

ret = re.match("[a-zA-Z0-9_]{8,20}","askfbahjsfyf")
print(ret.group())

emails = ['2958126254@qq.com','18035036167@163.com','18035036167@163.com123']
for email in emails:
    ret = re.match(r"[\w]{4,20}@163\.com$",email)
    if ret:
        print(f"{email}符合要求")
    else:
        print(f"{email}不符合要求")

nums = ['8','78','08','100']

for num in nums:
    #ret = re.match(r"[1-9]?\d", num) 08的时候返回了0
    ret = re.match(r"[1-9]?\d$|100",num)
    if ret:
        print(f"{num}符合要求")
    else:
        print(f"{num}不符合要求")

for email in emails:
    ret = re.match("\w{4,20}+@(163|qq|126)\.com$",email)
    if ret:
        print(f"{email}符合要求")
    else:
        print(f"{email}不符合要求")

tels = ["13100001234","18912344321","10086","18000007777"]

for tel in tels:
    ret = re.match("1\d{9}[1-35689]",tel)
    if ret:
        print(f"{tel}符合要求")
    else:
        print(f"{tel}不符合要求")

ret = re.match("([^-]+)-(\d+)","010-12345678")
print(ret.group(1))
print(ret.group(2))

ret = re.match(r"<([a-zA-Z]*)>\w*</\1>","<html>hh</html>")
print(ret.group())

labels = ["<html><h1>www.cskaoyan.com</h1></html>","<html><h1>www.cskaoyan.com</h2></html>"]

for label in labels:
    ret = re.match(r"<(\w*)><(\w*)>.*</\2></\1>",label)#.匹配除\n以外的任何字符，*表示0或者多个
    if ret:
        print(f"{label}符合要求")
    else:
        print(f"{label}不符合要求")

for label in labels:
    ret = re.match(r"<(?P<name1>\w*)><(?P<name3>\w*)>.*</(?P=name3)></(?P=name1)>",label)#.匹配除\n以外的任何字符，*表示0或者多个
    if ret:
        print(f"{label}符合要求")
    else:
        print(f"{label}不符合要求")