# 作者: 王梓豪
# 2025年06月03日23时27分11秒
# 2958126254@qq.com

class my_Stack:
    def __init__(self):
        self.stack = []
    def push(self,item):
        self.stack.append(item)
    def pop(self):
        return self.stack.pop()
    def top(self):
        if self.empty():
            return "empty"
        return self.stack[-1]
    def empty(self):
        if len(self.stack) == 0:
            return True
        return False
    def size(self):
        return len(self.stack)

if __name__ == '__main__':
    my_stack = my_Stack()
    my_stack.push(10)
    my_stack.push(20)
    my_stack.push(30)
    my_stack.push(40)
    print(my_stack.top())
    print(my_stack.pop())
    print(my_stack.pop(),my_stack.top())
    print(my_stack.pop())
    print(my_stack.pop())
    print(my_stack.empty())