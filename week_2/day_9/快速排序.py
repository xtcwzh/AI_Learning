# 作者: 王梓豪
# 2025年06月05日20时37分38秒
# 2958126254@qq.com

import random

def partition(nums,left,right):
    k = left
    random_pos = random.randint(left,right)
    nums[random_pos],nums[right] = nums[right],nums[random_pos] #随机数优化，防止最坏情况的发生
    key = nums[right]
    for i in range(left,right):
        if nums[i] < key:
            nums[i],nums[k] = nums[k],nums[i]
            k += 1
    nums[k],nums[right] = nums[right],nums[k]
    return k


def quick_sort(nums,left,right):
    if len(nums) <= 1:
        return nums
    if left < right:
        key = partition(nums,left,right)
        quick_sort(nums,left , key - 1)
        quick_sort(nums , key + 1,right)

nums = [20,3,5,124,86,1,12,5,7,9,89,23,435]
quick_sort(nums,0,len(nums)-1)
print(nums)