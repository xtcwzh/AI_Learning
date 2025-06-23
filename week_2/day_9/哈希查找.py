# 作者: 王梓豪
# 2025年06月05日23时12分33秒
# 2958126254@qq.com

MAXKEY = 1000

def elf_hash(hash_str):

    h = 0
    g = 0
    for i in hash_str:
        h = (h << 4) + ord(i)
        g = h & 0xf0000000
        if g:
            h ^= g >> 24
        h &= ~g
    return h % MAXKEY

str_list = ["xiongda", "lele", "hanmeimei", "wangdao", "fenghua"]
hash_table=[None]*MAXKEY  #初始化一个哈希表
for i in str_list:
    if hash_table[elf_hash(i)] is None:
        hash_table[elf_hash(i)]=[i] #第一次放入
    else:
        hash_table[elf_hash(i)].append(i) #哈希冲突后拉链法解决