import numpy as np


class Autopilot:
    def __init__(self, res=(640, 480), armed=False, PID=[0.4, 0, 0.4], auto_pilot_speed=25, apriltag_factor=1):
        self.resolution = res
        self.auto_pilot_armed = armed
        self.rc_params = (0, 0, 0, 0)
        self.PID = PID
        self.error = (0, 0, 0, 0)
        self.prev_error = (0, 0, 0, 0)
        self.autopilot_speed = auto_pilot_speed
        self.apriltag_final_area_factor = apriltag_factor
        self.apriltag_final_area = (self.resolution[0] * self.resolution[1] * 0.15 ** 2) * (
                    1 / self.apriltag_final_area_factor)

    def __repr__(self):
        return f'''
res: {self.resolution}
autopilot_armed: {self.auto_pilot_armed}
rc_params: {self.rc_params}
PID: {self.PID}
error: {self.error}
prev_error: {self.prev_error}
autopilot_speed: {self.autopilot_speed}
apriltag_final_area_factor: {self.apriltag_final_area_factor}
'''

    def arm_autopilot(self):
        self.auto_pilot_armed = True

    def disarm_autopilot(self):
        self.auto_pilot_armed = False

    def set_autopilot_speed(self, autopilot_speed):
        self.autopilot_speed = autopilot_speed

    def track_target(self, rc_params, target_xy, error, prev_error, target_area=None):
        self.rc_params = rc_params
        self.error = error
        self.prev_error = prev_error

        lr, fb, ud, yv = rc_params
        [cx, cy] = target_xy

        # finding landing spot at constant height
        area = target_area if target_area is not None else 0

        prp, itg, dif = self.PID
        lr_error, fb_error, ud_error, _ = error
        lr_error_prev, fb_error_prev, ud_error_prev, _ = prev_error

        # no ud movement if no target_area is provided
        if target_area == 0:
            ud_error = 0

        if cx is None or cx == 0:
            return (lr, fb, ud, yv)

        else:
            # rc commamnds range from -100 to 100
            # that is why errors below are scaled to 100

            # forwards/backwards
            fb = prp * fb_error + dif * (fb_error - fb_error_prev)
            fb = int(np.clip(fb, -self.autopilot_speed,
                             self.autopilot_speed))  # when error is large, limit the rc command to 25

            # left/right
            lr = prp * lr_error + dif * (lr_error - lr_error_prev)
            lr = int(np.clip(lr, -self.autopilot_speed, self.autopilot_speed))

            # up/down
            ud = prp * ud_error + dif * (ud_error - ud_error_prev)
            ud = int(np.clip(ud, -self.autopilot_speed, self.autopilot_speed))

        return (lr, fb, ud, yv)

    def get_alignment_error(self, target_center, target_area):
        [cx, cy] = target_center
        area = target_area
        width, height = self.resolution

        error = (0, 0, 0, 0)

        if cx is not None:
            # horizontal error
            lr_error = +1 * (cx - width / 2) / (width / 2) * 100  # -100 when tag on left, +100 when tag on right
            fb_error = -1 * (cy - height / 2) / (height / 2) * 100  # -100 when tag on top, +100 when tag on bottom

            # vertical error, not limited to 0-100 as area can be infinitely small
            ud_error = +1 * (area - self.apriltag_final_area) / (
                self.apriltag_final_area) * 100  # +100 when tag is very small, -100 when tag is very large

            error = (lr_error, fb_error, ud_error, 0)

        return error
