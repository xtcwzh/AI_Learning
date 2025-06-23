# 作者: 王梓豪
# 2025年06月03日21时56分22秒
# 2958126254@qq.com

import sys

print(sys.argv)

def w_file():
    file = open('Readme.md', 'a+',encoding='utf-8')
    file.write('王梓豪大王')
    txt = file.read()
    print(txt)
    file.close()

file = open('file1', 'a+',encoding='utf-8')
file.write('i an file1')
file.close()
file = open('file2', 'a+',encoding='utf-8')
file.write('i an file2')
file.close()
#w_file()
file =open(sys.argv[1])
print(file.read())
file.close()
file = open(sys.argv[2])
print(file.read())
#print(sys.argv[1])

