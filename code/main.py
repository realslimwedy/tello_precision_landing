
import pygame, cv2, time, sys, asyncio
from djitellopy import tello
import tello_package as tp


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
    
    video_feed_on = True
    height, width = 360, 240
    
    async def keyboard_rc():
        global me
        while True:
            # Control drone via keyboard
            rc_values = tp.keyboard_rc(me, (0, 0, 0, 0))
            me.send_rc_control(rc_values[0], rc_values[1], rc_values[2], rc_values[3])
            await asyncio.sleep(0.05)
            
            # Escape shuts down everything
            if tp.exit_app(me):
                me.land()
                me.streamoff()
                cv2.destroyAllWindows()
                pygame.quit()
                tp.connect_to_wifi("Leev_Marie")
                sys.exit()
            
    async def video_feed(video_feed_on):
        global me
        while True:
            if tp.video_feed_key_pressed(video_feed_on):
                if not video_feed_on:
                    video_feed_on = True
                elif video_feed_on:
                    video_feed_on = False
            if video_feed_on:
                img = me.get_frame_read().frame
                img_vid_feed = cv2.resize(img, (height, width))
                img_vid_feed = cv2.cvtColor(img_vid_feed, cv2.COLOR_BGR2RGB)
                cv2.imshow('Video Feed', img_vid_feed)
                cv2.waitKey(1) # this is necessary to show the image
            elif not video_feed_on:
                cv2.destroyAllWindows()
            
            await asyncio.sleep(0.05)
    
    async def image_saving():
        global me
        while True:
            if tp.save_image_key_pressed():
                img=me.get_frame_read().frame
                tp.save_image(img)
            await asyncio.sleep(0.2)


    await asyncio.gather(keyboard_rc(),image_saving(), video_feed(video_feed_on))

asyncio.run(main())

