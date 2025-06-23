# 作者: 王梓豪
# 2025年06月03日23时21分46秒
# 2958126254@qq.com

import os

def scan_dir(path,width):
    list_file = os.listdir(path)
    for file in list_file:
        print(' ' * width + file)
        new_path = os.path.join(path,file)
        if os.path.isdir(new_path):
            scan_dir(new_path,width + 4)

scan_dir('.',0)