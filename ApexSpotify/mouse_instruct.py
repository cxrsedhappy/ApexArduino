import hid
import time

MOUSE_LEFT = 0x01
MOUSE_RIGHT = 0x02
MOUSE_MIDDLE = 0x04
MOUSE_ALL = MOUSE_LEFT | MOUSE_RIGHT | MOUSE_MIDDLE


class DeviceNotFoundError(Exception):
    pass


class MouseInstruct:
    def __init__(self, device):
        self._buttons_mask = 0x0
        self._device = device
        self.move(0, 0)

    @classmethod
    def get_mouse(cls, vid=None, pid=None, ping_code=None):
        device = find_mouse_device(vid, pid, ping_code)
        if not device:
            vid_str = hex(vid) if vid else "Unspecified"
            pid_str = hex(pid) if pid else "Unspecified"
            ping_code_str = hex(ping_code) if pid else "Unspecified"
            error_msg = f"[-] Device VID: {vid_str} PID: {pid_str} Ping code: {ping_code_str} not found!"
            raise DeviceNotFoundError(error_msg)
        return cls(device)

    def _buttons(self, buttons):
        if buttons != self._buttons_mask:
            self._buttons_mask = buttons
            self.move(0, 0)

    def _make_report(self, x, y):
        report_data = [
            0x01, self._buttons_mask,
            x & 0xFF, (x >> 8) & 0xFF,
            y & 0xFF, (y >> 8) & 0xFF
        ]
        return report_data

    def _send_raw_report(self, report_data):
        self._device.write(report_data)

    def click(self, button=MOUSE_LEFT):
        self._buttons_mask = button
        self.move(0, 0)
        self._buttons_mask = 0
        self.move(0, 0)

    def silent_flick(self, x, y, button=MOUSE_LEFT):
        self._buttons_mask = button
        self.move(x, y)
        time.sleep(0.006)
        self._buttons_mask = 0
        self.move(-x, -y)

    def press(self, button=MOUSE_LEFT):
        self._buttons(self._buttons_mask | button)

    def release(self, button = MOUSE_LEFT):
        self._buttons(self._buttons_mask & ~button)

    def is_pressed(self, button = MOUSE_LEFT):
        return bool(button & self._buttons_mask)

    def move(self, x, y):
        self._send_raw_report(self._make_report(limit_xy(x), limit_xy(y)))


def check_ping(dev, ping_code):
    dev.write([0, ping_code])
    try:
        resp = dev.read(max_length=1, timeout_ms=10)
    except OSError as e:
        return False
    else:
        return resp and resp[0] == ping_code

def find_mouse_device(vid, pid, ping_code):
    device = hid.device()
    for dev_info in hid.enumerate(vid, pid):
        device.open_path(dev_info['path'])
        found = check_ping(device, ping_code)
        if found:
            return device
        else:
            device.close()
    return None

def low_byte(x):
    return x & 0xFF

def high_byte(x):
    return (x >> 8) & 0xFF

def limit_xy(xy):
    if xy < -32767:
        return -32767
    elif xy > 32767:
        return 32767
    else: return xy