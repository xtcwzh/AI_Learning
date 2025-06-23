# 作者: 王梓豪
# 2025年06月05日21时29分01秒
# 2958126254@qq.com

def adjust_max_heap(nums,adjust_pos,adjust_len):
    dad = adjust_pos
    son = 2 * dad + 1
    while son < adjust_len:
        if son + 1 < adjust_len and nums[son] < nums[son + 1]:
            son += 1
        if nums[dad] < nums[son]:
            nums[dad], nums[son] = nums[son],nums[dad]
            dad = son
            son = 2 * dad + 1
        else:
            break

def heap_sort(nums):
    length = len(nums)
    for dad_pos in range(length // 2 - 1, -1,-1):
        adjust_max_heap(nums,dad_pos,length)

    for i in range(length - 1, 0 , -1):
        #print(nums[0] ,end=' ')
        nums[0],nums[i] = nums[i],nums[0]
        adjust_max_heap(nums,0,i)

nums = [20,3,5,124,86,1,12,5,7,9,89,23,435]
heap_sort(nums)
print(nums)


