# 作者: 王梓豪
# 2025年06月03日21时36分41秒
# 2958126254@qq.com
def my_try():
    while True:
        try:
            num = input()
            try:
                num1 = int(num)
            except ValueError:
                raise ValueError(f"{num} is not an integer")
            num2 = str(num)
            if num2 == num2[::-1]:
                print(f"{num2} 是一个对称数")
            else:
                print(f"{num2} 不是一个对称数")
                raise ValueError(f"{num2}不是一个对称数")
        except ValueError as ve:
            print(f"错误捕获: {ve} 请重新输入。")
        except Exception as e:
            print(f"发生未知错误: {e}。请重新输入。")


my_try()