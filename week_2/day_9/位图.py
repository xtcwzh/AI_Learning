# 作者: 王梓豪
# 2025年06月05日23时33分40秒
# 2958126254@qq.com

def bit_map(my_list):
    int_bit_map = 0
    min_num = min(my_list)
    result = []
    for i in my_list:
        flag = 1 << (i - min_num)
        if flag & int_bit_map:
            pass
        else:
            int_bit_map = flag | int_bit_map
            result.append(i)
    return result


if __name__ == '__main__':
    list1 = [95, 17, 3, 31, 86, 75, 56, 19, 38, 26,
             94, 54, 53, 72, 59, 61, 74, 58, 78, 60,
             64, 43, 52, 90, 84, 19, -92, 2, 71, 12,
             67, 10, 53, 85, 98, 24, 11, 41, 44, 55,
             10, 47, 43, 98, 9, 55, 18, 30, 44, 22,
             48, 15, 87, 28, 47, 18, -92, 3, 38, 87, 59,
             84, 76, 65, 82, 26, 47, 52, 58, 79, 50, 82,
             5, 71, 28, 30, 17, 51, 11, 58, 12, 54, 49, 73,
             24, 46, 99, 94, 93, 70, 12, 33, 19, 67, 62, 74, 61, 89, 91, 51]
    print(len(bit_map(list1)))
    print(len(set(list1)))