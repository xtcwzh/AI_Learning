# 作者: 王梓豪
# 2025年06月04日22时26分10秒
# 2958126254@qq.com

from collections import deque
my_queue = deque([1, 2, 3])
print(my_queue)

class node:
    def __init__(self, value,left,right):
        self.value = value
        self.left = left
        self.right = right

class Tree:
    def __init__(self):
        self.root = None
        self.queue = deque()

    def insert(self,value):
        new_node = node(value,None,None)
        self.queue.append(new_node)
        if self.root is None:
            self.root = new_node
        else:
            if self.queue[0].left is None:
                self.queue[0].left = new_node
            else:
                self.queue[0].right = new_node
                self.queue.popleft()

    def pre_order(self,node):
        if node:
            print(node.value,end=" ")
            self.pre_order(node.left)
            self.pre_order(node.right)

    def mid_order(self,node):
        if node:
            self.mid_order(node.left)
            print(node.value,end=" ")
            self.mid_order(node.right)

    def post_order(self,node):
        if node:
            self.post_order(node.left)
            self.post_order(node.right)
            print(node.value,end=" ")

    def level_order(self,node):
        if not node:
            print("空树")
            return
        wd_queue = deque()
        wd_queue.append(node)
        while wd_queue:
            if wd_queue[0].left:
                wd_queue.append(wd_queue[0].left)
            if wd_queue[0].right:
                wd_queue.append(wd_queue[0].right)
            print(wd_queue[0].value,end=" ")
            wd_queue.popleft()

if __name__ == '__main__':
    tree = Tree()
    for i in range(1, 10):
        tree.insert(i)
    tree.pre_order(tree.root)
    print('\n------------------------')
    tree.mid_order(tree.root)
    print('\n------------------------')
    tree.post_order(tree.root)
    print('\n------------------------')
    tree.level_order(tree.root)
