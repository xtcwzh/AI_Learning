# 作者: 王梓豪
# 2025年06月05日22时59分13秒
# 2958126254@qq.com
from xxsubtype import bench


def bsearch(nums,target):
    low = 0
    high = len(nums) - 1
    while low <=high:
        mid = (low + high) // 2
        if target == nums[mid]:
            return mid
        if target < nums[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return -1

nums = [3, 10, 21, 29,33, 55, 74, 84, 84, 91]
print(bsearch(nums,74))

