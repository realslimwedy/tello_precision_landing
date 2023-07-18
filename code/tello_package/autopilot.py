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

    lr_error, fb_error, ud_error, yv_error = error
    lr_error_prev, fb_error_prev, ud_error_prev, yv_error_prev = prev_error
    prp, itg, dif = pid

    if cx==0:
        return (lr, fb, ud, yv)
    
    else:
        # forwards/backwards
        fb = prp * fb_error + dif*(fb_error - fb_error_prev)
        fb = int(np.clip(fb, -25, 25))

        # left/right
        lr = prp * lr_error + dif*(lr_error - lr_error_prev)
        lr = int(np.clip(lr, -25, 25))
        
        # up/down
        ud = prp * ud_error + dif*(ud_error - ud_error_prev)
        ud = int(np.clip(ud, -25, 25))

    return (lr, fb, ud, yv)
        


def get_alignment_error(apriltag_center, apriltag_area,ud_approach_center,res):
    pass
    [cx, cy] = apriltag_center
    area = apriltag_area
    width, height = res

    error = (0, 0, 0, 0)

    if cx!=0:
        lr_error = (cx - width/2) / (width/2)*100
        fb_error = (cy - height/2) / (height/2)*100
        ud_error =-1 * (area-ud_approach_center)/(ud_approach_center)*100
        error = (lr_error, fb_error, ud_error, 0)
    
    return error