# 作者: 王梓豪
# 2025年06月05日23时33分31秒
# 2958126254@qq.com

def count_sort(nums):
    cnt = [0] * 100
    res = []
    for num in nums:
        cnt[num] += 1
    cnt_1 = 0
    for i in range(len(cnt)):
        while cnt[i]:
            nums[cnt_1] = i
            cnt_1 += 1
            cnt[i] -= 1


nums = [1,6,43,65,87,1,98,3,6,9,23,4,19,87]
count_sort(nums)
print(nums)