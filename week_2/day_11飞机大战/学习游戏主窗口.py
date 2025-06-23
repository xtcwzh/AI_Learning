# 作者: 王梓豪
# 2025年06月07日23时57分11秒
# 2958126254@qq.com
import pygame
import time
pygame.init()
screen = pygame.display.set_mode((480,700)) #注意size参数的设置是(width,height)

bg = pygame.image.load("./images/background.png")
screen.blit(bg,(0,0))

hero = pygame.image.load("./images/me1.png")
screen.blit(hero,(200,500))

pygame.display.update()

clock = pygame.time.Clock()
#初始化矩型窗口
hero_rect = pygame.Rect(200,500,102,126)
while True:
    clock.tick(60)
    hero_rect.y -= 1
    if hero_rect.bottom <= 0:
        hero_rect.y = 700
    screen.blit(bg,(0,0))
    screen.blit(hero,hero_rect)
    pygame.display.update()

    event_list = pygame.event.get()
    if event_list:
        print(event_list)
        for event in event_list:
            if event.type == pygame.QUIT:
                print("游戏结束")
                pygame.quit()
                exit()

pygame.quit()