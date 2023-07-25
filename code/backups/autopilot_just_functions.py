import numpy as np


def autopilot_rc(me, flight_phase):
    if flight_phase == "approach":
        pass        
    elif flight_phase == "landing":
        pass


def track_apriltag(rc_params, apriltag_center, apriltag_area, pid, error, prev_error):
    lr, fb, ud, yv = rc_params
    [cx, cy] = apriltag_center
    area = apriltag_area
    prp, itg, dif = pid # PID= [0.4, 0, 0.4] 
    lr_error, fb_error, ud_error, _= error
    lr_error_prev, fb_error_prev, ud_error_prev, _ = prev_error
    

    if cx is None or cx == 0: # alternatively I could change the apriltag_module to return 0 if no tag is found
        return (lr, fb, ud, yv)
    
    else:
        # rc commamnds range from -100 to 100
        # that is why errors below are scaled to 100

        # forwards/backwards 
        fb = prp * fb_error + dif*(fb_error - fb_error_prev)
        fb = int(np.clip(fb, -25, 25)) # when error is large, limit the rc command to 25

        # left/right
        lr = prp * lr_error + dif*(lr_error - lr_error_prev)
        lr = int(np.clip(lr, -25, 25))
        
        # up/down
        ud = prp * ud_error + dif*(ud_error - ud_error_prev)
        ud = int(np.clip(ud, -25, 25))

    return (lr, fb, ud, yv)
        


def get_alignment_error(apriltag_center, apriltag_area, ud_approach_center_area,res):
    [cx, cy] = apriltag_center
    area = apriltag_area
    ud_approach_center = ud_approach_center_area
    width, height = res

    error = (0, 0, 0, 0)

    if cx is not None:
        
        # horizontal error
        lr_error = +1 * (cx - width/2) / (width/2)*100 # -100 when tag on left, +100 when tag on right
        fb_error = -1*(cy - height/2) / (height/2)*100 # -100 when tag on top, +100 when tag on bottom
        
        # vertical error, not limited to 0-100 as area can be infinitely small
        ud_error = +1 * (area-ud_approach_center)/(ud_approach_center)*100 # +100 when tag is very small, -100 when tag is very large
    
        error = (lr_error, fb_error, ud_error, 0)
    
    return error

