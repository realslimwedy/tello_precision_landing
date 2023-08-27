import numpy as np


class Autopilot:
    def __init__(self, res=(640, 480), PID=[0.4, 0, 0.4], auto_pilot_speed=25, apriltag_factor=1,
                 position_tolerance_threshold=10):
        self.resolution = res
        self.auto_pilot_armed = False
        self.rc_params = (0, 0, 0, 0)
        self.PID = PID
        self.error = (0, 0, 0, 0)
        self.prev_error = (0, 0, 0, 0)
        self.autopilot_speed = auto_pilot_speed
        self.apriltag_final_area_factor = apriltag_factor
        self.apriltag_final_area = (self.resolution[0] * self.resolution[1] * 0.15 ** 2) * (
                1 / self.apriltag_final_area_factor)
        self.position_tolerance_threshold = position_tolerance_threshold

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

    def get_alignment_error(self, center_of_target, target_area):
        [center_of_target_x, center_of_target_y] = center_of_target
        area_of_target = target_area
        width, height = self.resolution

        error = (0, 0, 0, 0)

        position_within_tolerance = False

        if center_of_target_x is not None:
            # horizontal error
            lr_error = +1 * (center_of_target_x - width / 2) / (
                        width / 2) * 100  # -100 when tag on left, +100 when tag on right
            fb_error = -1 * (center_of_target_y - height / 2) / (
                        height / 2) * 100  # -100 when tag on top, +100 when tag on bottom

            # vertical error, not limited to 0-100 as area_of_target can be infinitely small
            if area_of_target is not None:
                ud_error = +1 * (area_of_target - self.apriltag_final_area) / (
                    self.apriltag_final_area) * 100  # +100 when tag is very small, -100 when tag is very large
            else:
                ud_error = 0

            # check if center of target is within tolerance
            if abs(lr_error) < self.position_tolerance_threshold and abs(
                    fb_error) < self.position_tolerance_threshold and abs(ud_error) < self.position_tolerance_threshold:
                position_within_tolerance = True

            error = (lr_error, fb_error, ud_error, 0)

        return error, position_within_tolerance

    def track_target(self, rc_params, target_xy, error, prev_error, target_area=0):
        self.rc_params = rc_params
        self.error = error
        self.prev_error = prev_error

        lr, fb, ud, yv = rc_params
        [cx, cy] = target_xy

        # finding landing spot at constant height
        if target_area is not None:
            area = target_area
        else:
            area = 0
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
