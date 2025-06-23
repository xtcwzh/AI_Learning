# 作者: 王梓豪
# 2025年05月29日21时43分48秒
# 2958126254@qq.com

def find_two_unique_numbers(nums):
    xor_sum = 0
    for num in nums:
        xor_sum ^= num

    lowest_bit = xor_sum & (-xor_sum)
    unique_num1 = 0
    unique_num2 = 0

    for num in nums:
        if (num & lowest_bit) == 0:
            unique_num1 ^= num
        else:
            unique_num2 ^= num

    return unique_num1, unique_num2

my_array = [1, 2, 3, 4, 5, 3, 4, 5]

result = find_two_unique_numbers(my_array)
print(type(result))
print(f"数组 {my_array} 中出现一次的两个数是: {result}")


my_array2 = [-1, -1, -5, -5, 10, 20, 30, 30]
result2 = find_two_unique_numbers(my_array2)
print(f"数组 {my_array2} 中出现一次的两个数是: {result2}")