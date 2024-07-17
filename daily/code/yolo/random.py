import cv2
import numpy as np
import win32api
import win32con
import win32gui
from PIL import ImageGrab
from ultralytics import YOLO
import threading
# import ctypes # 导入罗技dll
import torch
import pydirectinput
import keyboard
import time
import math

running = False


def toggle_running():
    global running
    running = not running
    if running:
        print("开始运行.")
    else:
        print("停止.")


keyboard.on_press_key('shift', lambda _: toggle_running())
# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(r"E:\code\YOLO\ultralytics-main\my_model\aimlab\weights\best.pt").to(device)
# driver = ctypes.CDLL(r'E:\code\YOLO\ultralytics-main\MouseControl.dll')  # 罗技驱动
window_title = "aimlab_tb"


# 获取屏幕尺寸


# gw版
# def capture_screen(window_title):
#     window = gw.getWindowsWithTitle(window_title)[0]
#     bbox = (window.left, window.top, window.right, window.bottom)
#     img = ImageGrab.grab(bbox)
#     frame = np.array(img)
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     return frame

# win32版
def capture_screen(window_title):
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd == 0:
        print("No window found")
        return None  # 如果没有找到窗口，返回None
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    bbox = (left, top, right, bottom)
    img = ImageGrab.grab(bbox)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


class BoxInfo:
    def __init__(self, box):
        self.box = box  # 框的坐标 (x1, y1, x2, y2)


# def get_window_rect():
#     hwnd = win32gui.GetForegroundWindow()
#     if hwnd == 0:
#         raise Exception("No active window found.")
#     rect = win32gui.GetWindowRect(hwnd)
#     return rect

# 罗技鼠标移动
# def mouse_move(offset_x, offset_y):
#     driver.move_R(offset_x, offset_y)


def run_detection():
    cnt = 0
    direction = 1
    global running, frame
    global controlling_mouse, mouse_position, target_position
    while True:

        frame = capture_screen(window_title)
        if running:
            cnt += 1
            if cnt % 100 == 0:
                direction*=-1
            results = model.predict(source=frame)  # 预测结果
            # 存储所有检测到的框信息和距离
            targets = []
            for result in results:
                boxes = result.boxes  # 获取所有框的坐标
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # 计算框中心到鼠标的距离
                    target_x = (x1 + x2) // 2
                    target_y = (y1 + y2) // 2
                    targets.append((target_x, target_y, conf))
            targets.sort(key=lambda x: x[0] * x[1]*direction)  # 根据位置

            # 如果找到最近框，且在安全范围内，移动鼠标
            if targets:
                mouse_pos = win32api.GetCursorPos()
                target_x, target_y, _ = targets[0]
                # 计算偏移量
                offset_x = target_x - mouse_pos[0]
                offset_y = target_y - mouse_pos[1]
                # 移动
                pydirectinput.moveRel(offset_x, offset_y, relative=True)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                # 绘制所有检测框和中心点
                for target in targets:
                    x_center = target[0]
                    y_center = target[1]
                    cv2.circle(frame, (x_center, y_center), 5, (255, 0, 0), -1)
                    if target == targets[0]:
                        cv2.circle(frame, (x_center, y_center), 10, (0, 0, 255), -1)
        mouse_pos = win32api.GetCursorPos()
        cv2.circle(frame, (mouse_pos[0], mouse_pos[1]), 5, (255, 255, 0), -1)
        cv2.imshow("Screen Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cv2.destroyAllWindows()
    keyboard.unhook_all()


thread = threading.Thread(target=run_detection)
thread.start()
