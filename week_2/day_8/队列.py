# 作者: 王梓豪
# 2025年06月04日21时12分08秒
# 2958126254@qq.com

from collections import deque

my_queue = deque(["wzh","dwzh","ddwzh"])

my_queue.appendleft("wzh!!!")
print(my_queue)
my_queue.append("rwzh")
print(my_queue)
my_queue.popleft()
print(my_queue)
my_queue.pop()
print(my_queue)
my_queue[1] = "lyr"
print(my_queue)