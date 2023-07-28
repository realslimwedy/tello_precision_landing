import pygame
import cv2 as cv


screen_variables_names_units = {
    'names': {'battery_level': 'Battery Level', 'flight_phase': 'Flight Phase',
              'auto_pilot_armed': 'Auto-Pilot Armed',
              'speed': 'Speed', 'temperature': 'Temperature', 'flight_time': 'Flight Time'},
    'units': {'battery_level': '%', 'flight_phase': '', 'auto_pilot_armed': '', 'speed': '',
              'temperature': '°C',
              'flight_time': 'sec'}}


class Pygame():
    def __init__(self, res=(400, 400)):
        pygame.init()
        win = pygame.display.set_mode(res)
        self.screen = pygame.display.set_mode(res)
        pygame.display.set_caption("Drone Control")
        self.screen_variables_names_units = screen_variables_names_units

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

    def set_timer(self, event_id, time_ms):
        pygame.time.set_timer(event_id, time_ms)

    def display_multiple_status(self, screen, v_pos=10, h_pos=10, **kwargs):
        self.screen = screen
        self.v_pos = v_pos
        self.h_pos = h_pos
        self.kwargs = kwargs

        font = pygame.font.SysFont(None, 25)
        red = (255, 0, 0)
        orange = (255, 165, 0)

        for variable, value in kwargs.items():
            bg_color = (0, 0, 0)
            show_warning = None

            name = self.screen_variables_names_units['names'].get(variable)
            unit = self.screen_variables_names_units['units'].get(variable)

            if variable == "battery_level":
                show_warning = value <= 20
                bg_color = red
            elif variable == "auto_pilot_armed":
                show_warning = value == True
                bg_color = orange
            elif variable == "flight_phase":
                if value == "Approach":
                    show_warning = True
                    bg_color = orange
                elif value == "Landing":
                    show_warning = True
                    bg_color = red
            elif variable == "temperature":
                if value > 85:
                    show_warning = True
                    bg_color = orange
                elif value > 90:
                    show_warning = True
                    bg_color = red

            elif variable == "speed":
                show_warning = (value > 50)
                bg_color = orange

            status_text = font.render(f'{name}: {value} {unit}', True, (255, 255, 255))

            if variable == "speed" and value == 50:  # add empty space to avoid orange artifact
                status_text = font.render(f'{name}: {value}   {unit}', True, (255, 255, 255))
            else:
                status_text = font.render(f'{name}: {value} {unit}', True, (255, 255, 255))

            text_surface = pygame.Surface((status_text.get_width(), status_text.get_height()))

            if show_warning:
                text_surface.fill(bg_color)
            text_surface.blit(status_text, (0, 0))

            screen.blit(text_surface, (v_pos, h_pos))
            h_pos += 30


if __name__ == "__main__":
    pygame_instance = Pygame()
    show_warning = False
    warning_message = ""

    while True:

        if pygame_instance.get_key('ESCAPE'):
            pygame.quit()

        # Check if the 'w' key is pressed to toggle the warning message
        if pygame_instance.get_key('w'):
            show_warning = not show_warning
            if show_warning:
                warning_message = "This is a warning message!"
            else:
                warning_message = ""

        # Check if the 'a' key is pressed
        if pygame_instance.get_key('a'):
            print("Key 'a' was pressed")

        # Display the warning message or remove it based on the show_warning variable
        pygame_instance.display_status(pygame_instance.screen, warning_message, show_warning=show_warning)

        # Update the display
        pygame.display.update()

        # Add a small delay to prevent high CPU usage
        pygame.time.delay(50)
