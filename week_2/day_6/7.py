# 作者: 王梓豪
# 2025年06月02日23时29分17秒
# 2958126254@qq.com

class MusicPlayer:
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        print('音乐播放器初始化')


player1 = MusicPlayer()
player2 = MusicPlayer()
pass