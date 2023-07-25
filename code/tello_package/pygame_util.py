import pygame
import cv2

class Pygame():
    def __init__(self):
        pygame.init()
        win = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("Drone Control")

    def get_key(self, key_name):
        ans = False
        for event in pygame.event.get():
            pass
        key_input = pygame.key.get_pressed()
        my_key = getattr(pygame, f'K_{key_name}')
        if key_input[my_key]:
            ans = True
        pygame.display.update()
        return ans

    def display_video_feed(self, screen, image):
        img=image
        screen = screen
        img = cv2.flip(img, 1)  # flip horizontally
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = pygame.surfarray.make_surface(img)
        screen.blit(img, (0, 0))


    def display_status(self, screen, text, v_position=10, h_position=10):
        self.screen = screen
        font = pygame.font.SysFont(None, 25)
        status_text = font.render(text, True, (255, 255, 255))
        screen.blit(status_text, (v_position, h_position))


if __name__ == "__main__":
    while True:
        pygame_instance = Pygame()  # Instantiate the Pygame class
        print(pygame_instance.get_key('e'))  # Call the get_key method on the instance

        # esc
        if pygame_instance.get_key('q'):
            break