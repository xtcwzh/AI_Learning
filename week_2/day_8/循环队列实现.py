# 作者: 王梓豪
# 2025年06月04日21时15分46秒
# 2958126254@qq.com
from sympy.physics.units import coulomb_constant


class CircleQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [0] * capacity
        self.front = 0
        self.rear = -1
        self.count = 0

    def is_full(self):
        if self.count == self.capacity:
            return True
        return False
    def is_empty(self):
        if self.count == 0:
            return True
        return False
    def enqueue(self, value):
        if self.is_full():
            print("队列已满")
            return
        else:
            self.rear = (self.rear + 1) % self.capacity
            self.queue[self.rear] = value
            self.count += 1
    def dequeue(self):
        if self.is_empty():
            print("队列已空")
            return
        else:
            self.front = (self.front + 1) % self.capacity
            self.count -= 1

    def __str__(self):
        if self.is_empty():
            return []
        else:
            result = []
            for i in range(self.count):
                index = (self.front + i) % self.capacity
                result.append(self.queue[index])
            return (f"队列元素{result},元素个数{self.count}，队头位置{self.front}，队尾位置{self.rear}")


if __name__ == '__main__':
    c = CircleQueue(5)
    c.enqueue(1)
    c.enqueue(2)
    c.enqueue(3)
    c.enqueue(4)
    c.enqueue(5)
    print(c)
    c.dequeue()
    print(c)
    print(c.is_empty())
    print(c.is_full())
