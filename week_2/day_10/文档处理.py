# 作者: 王梓豪
# 2025年06月06日22时54分59秒
# 2958126254@qq.com

from operator import itemgetter

class Hash_str(object):
    def __init__(self, len):
        self.hash_len = len
        self.hash_map = [None] * len

    def hash_elf_algorithm(self, str):
        h = 0
        g = 0
        for i in str:
            h = (h << 4) + ord(i)  # ASCII
            g = h & 0xf00000000
        if g:
            h ^= g >> 24
        h &= ~g
        return h % self.hash_len  # 取余使得其落在len范围

if __name__ == '__main__':

    hash1 = Hash_str(10000)
    f = open('Beauty and The Beast.txt')
    text = f.read()
    for i in text.replace('\n', ' ').split(' '):
        if i != '':
            hash1.hash_map[hash1.hash_elf_algorithm(i)] = [i, 0]
    for i in text.replace('\n', ' ').split(' '):
        if i != '' and i in hash1.hash_map[hash1.hash_elf_algorithm(i)]:
            hash1.hash_map[hash1.hash_elf_algorithm(i)][1] += 1
    f.close()
    list1 = []
    for i in hash1.hash_map:
        if i != None:
            list1.append(i)
    list2 = sorted(list1, key=itemgetter(1), reverse=True)
    for i in range(10):
        print(list2[i], end=' ')

