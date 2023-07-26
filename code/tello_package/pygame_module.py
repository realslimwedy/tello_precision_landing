import pygame
import cv2 as cv


class Pygame():
    def __init__(self, res=(400, 400)):
        pygame.init()
        win = pygame.display.set_mode(res)
        self.screen = pygame.display.set_mode(res)
        pygame.display.set_caption("Drone Control")

    def __repr__(self):
        return f'Pygame window \"Drone Control\" openend'

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
        img = image
        screen = screen
        img = cv.flip(img, 1)  # flip horizontally
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        img = pygame.surfarray.make_surface(img)
        screen.blit(img, (0, 0))

    def display_status(self, screen, text, show_warning=False):
        self.screen = screen
        font = pygame.font.SysFont(None, 25)

        if show_warning:
            status_text = font.render(text, True, (255, 0, 0))  # Red color (R, G, B)
        else:
            status_text = font.render(text, True, (255, 255, 255))  # White color for normal message

        text_rect = status_text.get_rect()
        screen_rect = screen.get_rect()
        text_rect.center = screen_rect.center
        screen.blit(status_text, text_rect)

    def display_multiple_status(self, screen, screen_variables_names_units, v_pos=10, h_pos=10, **kwargs):
        self.screen = screen
        self.screen_variables_names_units = screen_variables_names_units
        self.v_pos = v_pos
        self.h_pos = h_pos
        self.kwargs = kwargs

        font = pygame.font.SysFont(None, 25)
        red = (255, 0, 0)  # Red color for the warning background

        for variable, value in kwargs.items():
            name = screen_variables_names_units['names'].get(variable)
            unit = screen_variables_names_units['units'].get(variable)

            # Determine if the warning should be displayed for this variable (e.g., "temperature")
            show_warning = variable == "temperature"  # Replace "temperature" with the variable name that triggers the warning

            # Set the background color based on the show_warning condition
            bg_color = red if show_warning else (0, 0, 0)  # Use red background if show_warning is True, otherwise black

            status_text = font.render(f'{name}: {value} {unit}', True, (255, 255, 255))

            # Create a surface with the specified background color and blit the text on it
            text_surface = pygame.Surface((status_text.get_width(), status_text.get_height()))
            text_surface.fill(bg_color)
            text_surface.blit(status_text, (0, 0))

            # Blit the surface on the screen at the specified position
            screen.blit(text_surface, (v_pos, h_pos))
            h_pos += 30


if __name__ == "__main__":
    pygame_instance = Pygame()

    while True:

        # Check if the 'w' key is pressed to trigger the warning message
        if pygame_instance.get_key('w'):
            warning_message = "This is a warning message!"
            pygame_instance.display_status(pygame_instance.screen, warning_message, show_warning=True)

        # Update the display
        pygame.display.update()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Add a small delay to prevent high CPU usage
        pygame.time.delay(50)

