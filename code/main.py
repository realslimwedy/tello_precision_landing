
import pygame, cv2, time, sys, asyncio
from djitellopy import tello
import tello_package as tp

# Main program

if __name__ == "__main__":

    # Connect to TELLO drone wifi
    tp.connect_to_wifi("TELLO-9C7357")

    # Initialize drone & connect
    tp.init_keyboard_control()
    me = tello.Tello()
    me.connect()
    print(f'Battery Level: {me.get_battery()} %')
    time.sleep(0.5)

    me.streamon()

    async def main():
        async def keyboard_control():
            global me
            while True:
                # Control drone via keyboard
                rc_values = tp.keyboard_rc(me, (0, 0, 0, 0))
                me.send_rc_control(rc_values[0], rc_values[1], rc_values[2], rc_values[3])
                await asyncio.sleep(0.05)
                if tp.exit_app(me):
                    me.land()
                    me.streamoff()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    tp.connect_to_wifi("Leev_Marie")
                    sys.exit()
                
        async def image_saving():
            global me
            while True:
                # Save image upon pressing spacebar
                img=me.get_frame_read().frame
                tp.save_image(img)
                await asyncio.sleep(0.2)

        async def video_display():
            global me
            while True:
                # Display video feed
                img = me.get_frame_read().frame
                img_vid_feed = cv2.resize(img, (360, 240))
                img_vid_feed = cv2.cvtColor(img_vid_feed, cv2.COLOR_BGR2RGB)
                cv2.imshow("Image", img_vid_feed)
                await asyncio.sleep(0.05)

        await asyncio.gather(keyboard_control(),image_saving(), video_display())

    asyncio.run(main())

