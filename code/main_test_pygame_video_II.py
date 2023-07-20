import pygame
from pygame.locals import *
from pygame import camera
import cv2

pygame.init()
pygame.camera.init()
screen = pygame.display.set_mode((640, 480))
cam = pygame.camera.Camera(0, (640, 480))  # Use camera index 0
cam.start()

while True:
    image = cam.get_image()
    screen.blit(image, (0, 0))
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == QUIT:
            cam.stop()
            pygame.quit()
            exit()
