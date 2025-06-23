# 作者: 王梓豪
# 2025年06月12日23时36分37秒
# 2958126254@qq.com


import numpy as np
import time

# a = np.random.rand(1000000)
#
# sum_a = 0
# start = time.time()
# sum_a = np.sum(a)
# end = time.time()
# print(end - start)
#
# sum_b = 0
# start = time.time()
# for i in a:
#     sum_b += i
# end = time.time()
#
# print(end - start)

#代码，完成Numpy的数组和原生python的性能对比
#生成1000000个随机数，并求和
a = np.random.rand(1000000)

#使用Numpy求和
start = time.time()
sum_a = np.sum(a)
end = time.time()
print("Numpy求和耗时:", end - start)

#使用原生Python求和
start = time.time()
sum_b = 0
for i in a:
    sum_b += i
end = time.time()
print("原生Python求和耗时:", end - start)
