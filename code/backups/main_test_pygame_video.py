import pygame, cv2, time, sys, asyncio
from djitellopy import tello

import tello_package as tp
import object_tracking as ot

def display_status(text):
    font = pygame.font.SysFont(None, 25)
    status_text = font.render(text, True, (255, 255, 255))
    screen.blit(status_text, (10, 10))  # Adjust the position as needed

if __name__ == '__main__':

    width, height = 640, 480
    res = (width, height)

    # initialize the webcam feed
    cap = cv2.VideoCapture(0)

    # drone control
    tp.init_keyboard_control()
    screen = pygame.display.set_mode((640, 480))
    async def main():

        async def show_video_feed():
            global img, auto_pilot
            while True:
                img = cap.read()[1]

                img = cv2.resize(img, res)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.flip(img, -1)
                # rotate counter-clockwise
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                await asyncio.sleep(1. / 30)

                # Convert the NumPy array image to a Pygame surface
                pygame_img = pygame.surfarray.make_surface(img)

                screen.blit(pygame_img, (0, 0))  # Blit the video feed onto the Pygame window

                # Display the status of the auto_pilot variable
                display_status("Auto Pilot: On")

                pygame.display.flip()  # Update the display

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

        await asyncio.gather(show_video_feed())


    asyncio.run(main())
