# 作者: 王梓豪
# 2025年06月04日23时51分37秒
# 2958126254@qq.com

def sort_maopao(num):
    n = len(num)
    for i in range(n-1):
        swap_judge = False
        for j in range(n - 1 -i):
            if num[j] > num[j + 1]:
                num[j], num[j + 1] = num[j + 1], num[j]
                swap_judge = True

        if not swap_judge:
            return
num = [1,3,5,2,76,2,34,6,12,6,3,67]
sort_maopao(num)
print(num)
