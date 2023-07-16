import pygame

# Initialize Pygame
pygame.init()

# Set display mode
win = pygame.display.set_mode((400, 400))

def get_key(key_name):
    ans = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                ans = True
    pygame.display.update()
    return ans


while True:
    ans = get_key("ESCAPE")
    if ans:
        print("True")
