# 作者: 王梓豪
# 2025年06月05日23时33分20秒
# 2958126254@qq.com

def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left_list = nums[:mid]
    #right_list = nums[mid + 1 :]
    right_list = nums[mid:]
    sort_left =  merge_sort(left_list)
    sort_right = merge_sort(right_list)
    return combine(sort_left,sort_right)

def combine(left_list,right_list):
    i = j = 0
    res = []
    while i < len(left_list) and j < len(right_list):
        if(left_list[i] < right_list[j]):
            res.append(left_list[i])
            i += 1
        else:
            res.append(right_list[j])
            j += 1

    while i < len(left_list):
        res.append(left_list[i])
        i += 1

    while j < len(right_list):
        res.append(right_list[j])
        j += 1
    return res

my_list1 = [64, 34, 25, 12, 22, 11, 90]
print(f"原始列表 1: {my_list1}")
sorted_list1 = merge_sort(my_list1)
print(f"排序后列表 1: {sorted_list1}")