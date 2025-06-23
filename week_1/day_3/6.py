# 作者: 王梓豪
# 2025年05月28日19时00分43秒
# 2958126254@qq.com

def NumberOf1(n):
    count = 0
    while n&0xffffffff != 0:
        count += 1
        n = n & (n-1)
    return count

n = int(input())
print(NumberOf1(n))