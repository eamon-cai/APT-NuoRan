# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
import re
import cv2
import math
import serial
import ctypes
import random
import win32ui
import win32gui
import win32file
import pytweening
import onnxruntime
import numpy as np
import networkx as nx
import ctypes.wintypes as wintypes

from loguru import logger
from ctypes import windll
from copy import deepcopy


# -------------------------------------------定义一个用于截屏的类-------------------------------------------
class Capture:
    hwnd = None # 要截屏的窗口句柄
    x, y, w, h = None, None, None, None # 截屏区域的坐标和大小

    def __init__(self):
        self.cDC = None # 兼容设备上下文对象
        self.dcObj = None # 设备上下文对象
        self.wDC = None # 窗口设备上下文对象

    def Init(self, hwnd, w, h, x1, y1):
        self.hwnd = hwnd # 设置窗口句柄
        self.w, self.h, self.x1, self.y1 = w, h, x1, y1 # 设置截屏大小

        # 获取窗口和客户区的矩形
        window_rect = win32gui.GetWindowRect(hwnd)
        client_rect = win32gui.GetClientRect(hwnd)
        # 获取客户区左上角在屏幕坐标系中的位置
        left_corner = win32gui.ClientToScreen(hwnd, (0, 0))
        # 计算截屏区域相对于窗口矩形的坐标
        total_w = client_rect[2] - client_rect[0]
        total_h = client_rect[3] - client_rect[1]
        self.x = (total_w - w) // 2 + left_corner[0] - window_rect[0]
        self.y = (total_h - h) // 2 + left_corner[1] - window_rect[1]
        self.x = self.x - self.x1
        self.y = self.y - self.y1
        # 创建窗口和兼容位图的设备上下文对象
        self.wDC = win32gui.GetWindowDC(self.hwnd)
        self.dcObj = win32ui.CreateDCFromHandle(self.wDC)
        self.cDC = self.dcObj.CreateCompatibleDC()

    def InitEx(self, hwnd, x, y, w, h):
        self.hwnd = hwnd # 设置窗口句柄
        self.x, self.y, self.w, self.h = x, y, w, h # 设置截屏区域的坐标和大小
        # 创建窗口和兼容位图的设备上下文对象
        self.wDC = win32gui.GetWindowDC(self.hwnd)
        self.dcObj = win32ui.CreateDCFromHandle(self.wDC)
        self.cDC = self.dcObj.CreateCompatibleDC()

    def capture(self):
        try:
            dataBitMap = win32ui.CreateBitmap() # 创建一个位图对象
            dataBitMap.CreateCompatibleBitmap(self.dcObj, self.w, self.h) # 创建一个与截屏区域大小相同的兼容位图
            self.cDC.SelectObject(dataBitMap) # 将位图对象选择到兼容设备上下文中
            # 使用BitBlt函数将像素从窗口设备上下文复制到兼容设备上下文中
            self.cDC.BitBlt((0, 0), (self.w, self.h), self.dcObj, (self.x, self.y), 0x00CC0020)
            # 获取位图的字节数据作为一个字节数组
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            # 将字节数组转换为一个uint8类型的numpy数组，并重塑为与图像维度和通道相匹配的形状
            cut_img = np.frombuffer(signedIntsArray, dtype='uint8')
            cut_img.shape = (self.h, self.w, 4)
            cut_img = cut_img[..., :3] # 去除alpha通道
            win32gui.DeleteObject(dataBitMap.GetHandle()) # 删除位图对象并释放其资源
            cut_img = np.ascontiguousarray(cut_img) # 确保数组在内存中是连续的，以便与opencv兼容
            return cut_img # 返回截取的图像作为一个numpy数组
        except:
            logger.error('窗口句柄已消失，请结束进程重新捕获！！') # 如果出现错误，打印错误信息
            return None # 如果截屏失败，返回None

    def release_resource(self):
        win32gui.DeleteObject(self.wDC.GetHandle()) # 删除窗口设备上下文对象并释放其资源
        self.wDC, self.dcObj, self.cDC = None, None, None # 将设备上下文对象设置为None












# -------------------------------------------定义一个用于截屏的类-------------------------------------------







# -------------------------------------------定义一个用于随机的类-------------------------------------------





class HumanClicker():
    def __init__(self):
        pass
    def move_to(self, toPoint, humanCurve=None):
        fromPoint = [0, 0]
        if not humanCurve:
            humanCurve = HumanCurve(fromPoint,
                                    toPoint)

        extra_numbers = [0, 0]
        total_offset = [0, 0]
        origin = [0, 0]
        r_path = []
        for point in humanCurve.points:
            x_offset, y_offset = point[0] - origin[0], point[1] - origin[1]

            extra_numbers[0] += x_offset - int(x_offset)
            extra_numbers[1] += y_offset - int(y_offset)

            offset_to_append = [0, 0]
            if abs(extra_numbers[0]) > 1:
                offset_to_append[0] = int(extra_numbers[0])
                total_offset[0] += int(extra_numbers[0])
                extra_numbers[0] -= int(extra_numbers[0])
            if abs(extra_numbers[1]) > 1:
                offset_to_append[1] = int(extra_numbers[1])
                total_offset[1] += int(extra_numbers[1])
                extra_numbers[1] -= int(extra_numbers[1])
            r_path.append((int(x_offset), int(y_offset)))
            origin[0], origin[1] = point[0], point[1]
            total_offset[0] += int(x_offset)
            total_offset[1] += int(y_offset)
        total_offset[0] += int(extra_numbers[0])
        total_offset[1] += int(extra_numbers[1])
        return r_path


class BezierCurve():
    @staticmethod
    def binomial(n, k):
        return math.factorial(n) / float(math.factorial(k) * math.factorial(n - k))

    @staticmethod
    def bernsteinPolynomialPoint(x, i, n):
        return BezierCurve.binomial(n, i) * (x ** i) * ((1 - x) ** (n - i))

    @staticmethod
    def bernsteinPolynomial(points):
        def bern(t):
            n = len(points) - 1
            x = y = 0
            for i, point in enumerate(points):
                bern = BezierCurve.bernsteinPolynomialPoint(t, i, n)
                x += int(point[0] * bern)
                y += int(point[1] * bern)
            return x, y
        return bern

    @staticmethod
    def curvePoints(n, points):
        curvePoints = []
        bernstein_polynomial = BezierCurve.bernsteinPolynomial(points)
        for i in range(n):
            t = i / (n - 1)
            curvePoints += bernstein_polynomial(t),
        return curvePoints


def isNumeric(val):
    return isinstance(val, (float, int, np.int32, np.int64, np.float32, np.float64))

def isListOfPoints(l):
    if not isinstance(l, list):
        return False
    try:
        isPoint = lambda p: ((len(p) == 2) and isNumeric(p[0]) and isNumeric(p[1]))
        return all(map(isPoint, l))
    except (KeyError, TypeError) as e:
        return False



class HumanCurve():
    def __init__(self, fromPoint, toPoint, **kwargs):
        self.fromPoint = fromPoint
        self.toPoint = toPoint
        self.points = self.generateCurve(**kwargs)

    def generateCurve(self, **kwargs):
        offsetBoundaryX = 1
        offsetBoundaryY = 3
        leftBoundary = kwargs.get("leftBoundary", min(self.fromPoint[0], self.toPoint[0])) - offsetBoundaryX
        rightBoundary = kwargs.get("rightBoundary", max(self.fromPoint[0], self.toPoint[0])) + offsetBoundaryX
        downBoundary = kwargs.get("downBoundary", min(self.fromPoint[1], self.toPoint[1])) - offsetBoundaryY
        upBoundary = kwargs.get("upBoundary", max(self.fromPoint[1], self.toPoint[1])) + offsetBoundaryY
        tween = kwargs.get("tweening", pytweening.easeOutCirc)
        internalKnots = self.generateInternalKnots(int(leftBoundary), int(rightBoundary), int(downBoundary), int(upBoundary), 10)
        points = self.generatePoints(internalKnots)
        if abs(self.toPoint[0]) <= 150 and abs(self.toPoint[0]) >= 5 :
            points = self.tweenPoints(points, tween, 20)
        return points

    def generateInternalKnots(self, \
        leftBoundary, rightBoundary, \
        downBoundary, upBoundary,\
        knotsCount):
        knotsX = np.random.choice(range(leftBoundary, rightBoundary), size=knotsCount)
        knotsY = np.random.choice(range(downBoundary, upBoundary), size=knotsCount)
        knots = list(zip(knotsX, knotsY))
        return knots

    def generatePoints(self, knots):
        midPtsCnt = max( \
            abs(self.fromPoint[0] - self.toPoint[0]), \
            abs(self.fromPoint[1] - self.toPoint[1]), \
            2)
        knots = [self.fromPoint] + knots + [self.toPoint]

        return BezierCurve.curvePoints(midPtsCnt, knots)

    def distortPoints(self, points, distortionMean, distortionStdev, distortionFrequency):
        distorted = []
        for i in range(1, len(points)-1):
            x,y = points[i]
            delta = np.random.normal(distortionMean, distortionStdev) if \
                random.random() < distortionFrequency else 0
            distorted += (x,y+delta),
        distorted = [points[0]] + distorted + [points[-1]]
        return distorted

    def tweenPoints(self, points, tween, targetPoints):
        # tween is a function that takes a float 0..1 and returns a float 0..1
        res = []
        for i in range(targetPoints):
            index = int(tween(float(i)/(targetPoints-1)) * (len(points)-1))
            res += points[index],
        return res



# -------------------------------------------定义一个用于随机的类-------------------------------------------









# -------------------------------------------定义一个用于移动的类-------------------------------------------



class PID:
    """增量式PID控制算法"""

    def __init__(self, dim=1, Kp=5.0, Ki=0.001, Kd=10.0, Kaw=0.15, u_max=np.inf, u_min=-np.inf, max_err=np.inf):
        self.dim = dim  # 反馈信号y和跟踪信号v的维度

        # PID超参（不需要遍历的数据设置为一维数组）
        self.Kp = np.array(Kp).flatten()  # Kp array(dim,) or array(1,)
        self.Ki = np.array(Ki).flatten()  # Ki array(dim,) or array(1,)
        self.Kd = np.array(Kd).flatten()  # Kd array(dim,) or array(1,)
        self.Kaw = np.array(Kaw).flatten() / (self.Kd + 1e-8)  # Kaw取 0.1~0.3 Kd

        # 抗积分饱和PID（需要遍历的数据设置为一维数组，且维度保持和dim一致）
        self.u_max = np.array(u_max).flatten()  # array(1,) or array(dim,)
        self.u_max = self.u_max.repeat(self.dim) if len(self.u_max) == 1 else self.u_max  # array(dim,)
        self.u_min = np.array(u_min).flatten()  # array(1,) or array(dim,)
        self.u_min = self.u_min.repeat(self.dim) if len(self.u_min) == 1 else self.u_min  # array(dim,)
        self.max_err = np.array(max_err).flatten()  # array(1,) or array(dim,)
        self.max_err = self.max_err.repeat(self.dim) if len(self.max_err) == 1 else self.u_min  # array(dim,)


        # 控制器初始化
        self.u = np.zeros(self.dim)  # array(dim,)
        self.error_last = np.zeros(self.dim)  # array(dim,)
        self.error_last2 = np.zeros(self.dim)  # e(k-2)
        self.error_sum = np.zeros(self.dim)  # 这里integration是积分增量,error_sum是积分
        self.integration = np.zeros(self.dim)  # array(dim,)
        self.t = 0



    def gengxing(self, Kp, Ki, Kd):
        self.Kp = np.array(Kp).flatten()  # Kp array(dim,) or array(1,)
        self.Ki = np.array(Ki).flatten()  # Ki array(dim,) or array(1,)
        self.Kd = np.array(Kd).flatten()  # Kd array(dim,) or array(1,)


    def reset(self):
        # 控制器初始化
        self.u = np.zeros(self.dim)  # array(dim,)
        self.error_last = np.zeros(self.dim)  # array(dim,)
        self.error_last2 = np.zeros(self.dim)  # e(k-2)
        self.error_sum = np.zeros(self.dim)  # 这里integration是积分增量,error_sum是积分
        self.integration = np.zeros(self.dim)  # array(dim,)


    def update(self, y):

        # 计算PID误差
        error = y  # P偏差 array(dim,)

        # 抗积分饱和算法
        self.integration = np.zeros(self.dim)  # 积分增量 integration = error - 反馈信号
        beta = self._anti_integral_windup(error, method=2)  # 积分分离参数 array(dim,)

        K_p = self.Kp * (error - self.error_last)
        K_i = beta * self.Ki * self.integration
        K_d = self.Kd * (error - 2 * self.error_last + self.error_last2)
        # 控制量
        u0 = K_p + K_i + K_d

        self.u = u0 + self.u  # 增量式PID对u进行clip后有超调

        self.error_last2 = deepcopy(self.error_last)
        self.error_last = deepcopy(error)

        return -self.u

    # 抗积分饱和算法 + 积分分离
    def _anti_integral_windup(self, error, method=2):
        gamma = np.zeros(self.dim) if method < 2 else None  # 方法1的抗积分饱和参数
        # 积分分离，误差超限去掉积分控制
        beta = 0 if abs(error) > self.max_err else 1

        # 算法1
        if method < 2:
            # 控制超上限累加负偏差，误差超限不累加
            if self.u > self.u_max:
                if error < 0:
                    gamma = 1  # 负偏差累加
                else:
                    gamma = 0  # 正偏差不累加
            # 控制超下限累加正偏差，误差超限不累加
            elif self.u < self.u_max:
                if error > 0:
                    gamma = 1  # 正偏差累加
                else:
                    gamma = 0  # 负偏差不累加
            else:
                gamma = 1  # 控制不超限，正常累加偏差
            # end if
        # end if

        # 抗饱和算法1
        self.integration += error if method > 1 else beta * gamma * error  # 正常积分PID
        # self.integration += error/2 if method > 1 else beta * gamma * error/2 # 梯形积分PID

        # 反馈抑制抗饱和算法 back-calculation
        if method > 1:
            antiWindupError = np.clip(self.u, self.u_min, self.u_max) - self.u
            self.integration += self.Kaw * antiWindupError  # 累计误差加上个控制偏差的反馈量

        return beta

class ADRC(object):
    def __init__(self, w0, b0, w_n, sigma=1., time_delta=0.0001):
        self.u_p = 0
        self.x1_p = 0
        self.x2_p = 0
        self.x3_p = 0
        self.J = 0

        self.time_delta = time_delta
        self.b0 = b0

        self.b1 = 3 * w0
        self.b2 = 3 * w0**2
        self.b3 = w0**3

        self.kp = w_n * w_n
        self.kd = 2 * sigma * w_n

    def update(self, y, yref):
        err = y - self.x1_p

        x1_t = self.x1_p + self.time_delta * (self.x2_p + self.b1 * err)
        x2_t = self.x2_p + \
            self.time_delta * (self.x3_p + self.b0 * self.u_p + self.b2 * err)
        x3_t = self.x3_p + self.time_delta * self.b3 * err

        e = yref - x1_t

        self.J = self.J + e * e * self.time_delta
        u = (1/self.b0) * (e*self.kp - self.kd*x2_t - x3_t)

        self.x1_p, self.x2_p, self.x3_p = x1_t, x2_t, x3_t

        self.u_p = u

        return -u

    def gengxing(self, b0):
        self.b0 = b0


    def reset(self):
        self.u_p = 0
        self.x1_p = 0
        self.x2_p = 0
        self.x3_p = 0
        self.J = 0


# -------------------------------------------定义一个用于移动的类-------------------------------------------












# -------------------------------------------定义一个用于驱动的类-------------------------------------------




handle = 0
found = False

def _DeviceIoControl(devhandle, ioctl, inbuf, inbufsiz, outbuf, outbufsiz):

    DeviceIoControl_Fn = windll.kernel32.DeviceIoControl
    DeviceIoControl_Fn.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID,
                                   wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    DeviceIoControl_Fn.restype = wintypes.BOOL

    dwBytesReturned = wintypes.DWORD(0)
    lpBytesReturned = ctypes.byref(dwBytesReturned)
    status = DeviceIoControl_Fn(int(devhandle), ioctl, inbuf, inbufsiz, outbuf, outbufsiz, lpBytesReturned, None)
    return status, dwBytesReturned


class MOUSE_IO(ctypes.Structure):
    _fields_ = [
        ("button", ctypes.c_char),
        ("x", ctypes.c_byte),
        ("y", ctypes.c_byte),
        ("wheel", ctypes.c_char),
        ("unk1", ctypes.c_char),
    ]


def device_initialize(device_name):
    global handle
    try:
        handle = win32file.CreateFileW(device_name, win32file.GENERIC_WRITE, 0, None, win32file.OPEN_ALWAYS,
                                       win32file.FILE_ATTRIBUTE_NORMAL, 0)
    except:
        pass
    return bool(handle)


def mouse_open():
    global found
    global handle
    buffer0 = "\\??\\ROOT#SYSTEM#0002#{1abc05c0-c378-41b9-9cef-df1aba82b015}"
    status = device_initialize(buffer0)
    if status == True:
        found = True
    else:
        buffer1 = "\\??\\ROOT#SYSTEM#0001#{1abc05c0-c378-41b9-9cef-df1aba82b015}"
        status = device_initialize(buffer1)
        if status == True:
            found = True
    return found


def call_mouse(buffer):
    global handle
    return _DeviceIoControl(handle, 0x2a2010, ctypes.c_void_p(ctypes.addressof(buffer)), ctypes.sizeof(buffer), 0, 0)[
        0] == 0


def mouse_close():
    global handle
    win32file.CloseHandle(int(handle))
    handle = 0

if not mouse_open():
    print("Ghub没打开，或者可能有其他的问题！！")


def mouse_move(x, y):
    global handle

    while x > 127:
        io = MOUSE_IO()
        io.x = 127
        io.y = 0
        io.unk1 = 0
        io.button = 0
        io.wheel = 0
        if not call_mouse(io):
            mouse_close()
            mouse_open()
        x -= 127

    while x < -127:
        io = MOUSE_IO()
        io.x = -127
        io.y = 0
        io.unk1 = 0
        io.button = 0
        io.wheel = 0
        if not call_mouse(io):
            mouse_close()
            mouse_open()
        x += 127

    while y > 127:
        io = MOUSE_IO()
        io.x = 0
        io.y = 127
        io.unk1 = 0
        io.button = 0
        io.wheel = 0
        if not call_mouse(io):
            mouse_close()
            mouse_open()
        y -= 127

    while y < -127:
        io = MOUSE_IO()
        io.x = 0
        io.y = -127
        io.unk1 = 0
        io.button = 0
        io.wheel = 0
        if not call_mouse(io):
            mouse_close()
            mouse_open()
        y += 127

    io = MOUSE_IO()
    io.x = x
    io.y = y
    io.unk1 = 0
    io.button = 0
    io.wheel = 0
    if not call_mouse(io):
        mouse_close()
        mouse_open()



PUL = ctypes.POINTER(ctypes.c_ulong)


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("mi", MouseInput), ]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

ii_ = Input_I()
extra = ctypes.c_ulong(0)

def ctypes_moveR(x0, y0):
    ii_.mi = MouseInput(x0, y0, 0, 0x0001, 0, ctypes.pointer(extra))
    command = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

# -------------------------------------------定义一个用于驱动的类-------------------------------------------










# -------------------------------------------定义一个用于追踪的类-------------------------------------------


GENERATE_SET = 1
TERMINATE_SET = 3

class Matcher:
    def __init__(self):
        pass

    @classmethod
    def match(cls, state_list, measure_list):
        graph = nx.Graph()
        for idx_sta, state in enumerate(state_list):
            state_node = 'state_%d' % idx_sta
            graph.add_node(state_node, bipartite=0)
            for idx_mea, measure in enumerate(measure_list):
                mea_node = 'mea_%d' % idx_mea
                graph.add_node(mea_node, bipartite=1)
                score = cls.cal_iou(state, measure)
                if score is not None:
                    graph.add_edge(state_node, mea_node, weight=score)
        match_set = nx.max_weight_matching(graph)
        res = dict()
        for (node_1, node_2) in match_set:
            if node_1.split('_')[0] == 'mea':
                node_1, node_2 = node_2, node_1
            res[node_1] = node_2
        return res

    @classmethod
    def cal_iou(cls, state, measure):
        state = mea2box(state)
        measure = mea2box(measure)
        s_tl_x, s_tl_y, s_br_x, s_br_y = state[0], state[1], state[2], state[3]
        m_tl_x, m_tl_y, m_br_x, m_br_y = measure[0], measure[1], measure[2], measure[3]
        # 计算相交部分的坐标
        x_min = max(s_tl_x, m_tl_x)
        x_max = min(s_br_x, m_br_x)
        y_min = max(s_tl_y, m_tl_y)
        y_max = min(s_br_y, m_br_y)
        inter_h = max(y_max - y_min + 1, 0)
        inter_w = max(x_max - x_min + 1, 0)
        inter = inter_h * inter_w
        try:
            if inter == 0:
                return 0
            else:
                return inter/ ((s_br_x - s_tl_x) * (s_br_y - s_tl_y) + (m_br_x - m_tl_x) * (m_br_y - m_tl_y) - inter)
        except:
            return 0
def state2box(state):
    center_x = state[0]
    center_y = state[1]
    w = state[2]
    h = state[3]
    return [int(i) for i in [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]]

def box2meas(box):
    cls = box[0]
    x0 = box[1]
    y0 = box[2]
    x1 = box[3]
    y1 = box[4]
    return np.array([[cls, x0, y0, x1, y1]]).T

def mea2box(mea):
    center_x = mea[0]
    center_y = mea[1]
    w = mea[2]
    h = mea[3]

    return [int(i) for i in [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]]


def mea2state(mea):
    return np.row_stack((mea, np.zeros((2, 1))))




class Kalman:
    next_id = 0  # 添加新的类变量 next_id
    def __init__(self, A, B, H, Q, R, X, P):
        # 固定参数
        self.A = A  # 状态转移矩阵
        self.B = B  # 控制矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 过程噪声
        self.R = R  # 量测噪声
        # 迭代参数
        self.X_posterior = X  # 后验状态
        self.P_posterior = P  # 后验误差矩阵
        self.X_prior = None  # 先验状态
        self.P_prior = None  # 先验误差矩阵
        self.K = None  # kalman gain
        self.Z = None  # 观测
        # 起始和终止策略
        self.terminate_count = TERMINATE_SET
        # 赋予ID
        self.id = Kalman.next_id  # 为每个实例增加一个 id 属性
        Kalman.next_id += 1  # 将 next_id 递增 1

    def predict(self):
        self.X_prior = np.dot(self.A, self.X_posterior)
        self.P_prior = np.dot(np.dot(self.A, self.P_posterior), self.A.T) + self.Q
        return self.X_prior, self.P_prior

    @staticmethod
    def association(kalman_list, mea_list):
        # print(mea_list)
        # 记录需要匹配的状态和量测
        state_rec = {i for i in range(len(kalman_list))}
        mea_rec = {i for i in range(len(mea_list))}

        # 将状态类进行转换便于统一匹配类型
        state_list = list()  # [c_x, c_y, w, h].T
        for kalman in kalman_list:
            state = kalman.X_prior
            state_list.append(state[0:4])

        # 进行匹配得到一个匹配字典
        match_dict = Matcher.match(state_list, mea_list)

        # 根据匹配字典，将匹配上的直接进行更新，没有匹配上的返回
        state_used = set()
        mea_used = set()
        match_list = list()
        for state, mea in match_dict.items():
            state_index = int(state.split('_')[1])
            mea_index = int(mea.split('_')[1])
            match_list.append([state2box(state_list[state_index]), mea2box(mea_list[mea_index])])
            kalman_list[state_index].update(mea_list[mea_index])
            state_used.add(state_index)
            mea_used.add(mea_index)

        # 求出未匹配状态和量测，返回
        return list(state_rec - state_used), list(mea_rec - mea_used), match_list

    def update(self, mea=None):
        status = True
        if mea is not None:  # 有关联量测匹配上
            self.Z = mea

            self.K = np.dot(np.dot(self.P_prior, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P_prior), self.H.T) + self.R))  # 计算卡尔曼增益
            # print(self.H, self.X_prior)
            self.X_posterior = self.X_prior + np.dot(self.K, self.Z - np.dot(self.H, self.X_prior))  # 更新后验估计
            # print(np.eye(6))
            self.P_posterior = np.dot(np.eye(7) - np.dot(self.K, self.H), self.P_prior)  # 更新后验误差矩阵
            status = True
            self.Z = np.vstack((mea, self.id))
        else:  # 无关联量测匹配上
            if self.terminate_count == 1:
                status = False
            else:
                self.terminate_count -= 1
                self.X_posterior = self.X_prior
                self.P_posterior = self.P_prior
                status = True

        return status, self.X_posterior, self.P_posterior




class kalmanP():
    def __init__(self, GENERATE=1, TERMINATE=5):
        global GENERATE_SET, TERMINATE_SET
        GENERATE_SET = GENERATE  # 设置航迹起始帧数
        TERMINATE_SET = TERMINATE  # 设置航迹终止帧数
        # --------------------------------Kalman参数---------------------------------------
        # 状态转移矩阵，上一时刻的状态转移到当前时刻
        self.A = np.array([[1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]])

        # 控制输入矩阵B
        self.B = None
        # 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
        # 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
        self.Q = np.eye(self.A.shape[0]) * 500
        # 状态观测矩阵
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0]])

        # 观测噪声协方差矩阵R，p(v)~N(0,R)
        # 观测噪声来自于检测框丢失、重叠等
        self.R = np.eye(self.H.shape[0]) * 2
        # 状态估计协方差矩阵P初始化
        self.P = np.eye(self.A.shape[0])
        # -------------------------------------------------------------------------------

        self.state_list = []
        self.boxs = []
        self.boxs_copy = []

    def predict(self, boxs):
        # 预测
        for target in self.state_list:
            target.predict()
        # 关联
        array_list = [np.array(i) for i in boxs]
        mea_list = [box2meas(mea) for mea in array_list]
        state_rem_list, mea_rem_list, match_list = Kalman.association(self.state_list, mea_list)

        # 状态没匹配上的，更新一下，如果触发终止就删除
        state_del = list()
        for idx in state_rem_list:
            status, _, _ = self.state_list[idx].update()
            if not status:
                state_del.append(idx)
        self.state_list = [self.state_list[i] for i in range(len(self.state_list)) if i not in state_del]

        # 量测没匹配上的，作为新生目标进行航迹起始
        for idx in mea_rem_list:
            self.state_list.append(Kalman(self.A, self.B, self.H, self.Q, self.R, mea2state(mea_list[idx]), self.P))


        self.boxs.clear()
        for kalman in self.state_list:
            Z = kalman.Z  # 获取观测值
            self.boxs.append(Z)
        return self.boxs




# -------------------------------------------定义一个用于追踪的类-------------------------------------------











# -------------------------------------------定义一个用于推理的类-------------------------------------------



def vis(img, boxes, scores, cls_ids, conf=0.5): # 定义一个函数，名为vis，接受六个参数
    for i in range(len(boxes)): # 用一个循环遍历所有检测到的物体
        box = boxes[i] # 获取第i个物体的边框坐
        cls_id = int(cls_ids[i]) # 获取第i个物体的类别编号，并转换为整数
        score = scores[i] # 获取第i个物体的置信度
        if score < conf: # 如果置信度小于阈值
            continue # 跳过该物体，继续下一个循环
        x0 = int(box[0]) # 获取边框左上角的x坐标，并转换为整数
        y0 = int(box[1]) # 获取边框左上角的y坐标，并转换为整数
        x1 = int(box[2]) # 获取边框右下角的x坐标，并转换为整数
        y1 = int(box[3]) # 获取边框右下角的y坐标，并转换为整数
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist() # 根据类别编号从_COLORS列表中选择一种颜色，并乘以255，转换为无符号8位整数，再转换为列表
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2) # 在图片上绘制物体的边框，参数分别为图片、左上角坐标、右下角坐标、颜色和线宽
    return img # 返回修改后的图片数据

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):

    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])
def box_iou(box1, box2, eps=1e-7):
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clip(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].split(2, 2), box2.split(2, 0)
    inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)
def v5nms(boxes, scores, iou_threshold):
    # 将边界框转换为(x1, y1, x2, y2)格式
    boxes = xywh2xyxy(boxes)
    # 计算每个边界框的面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 按照得分降序排序，获取排序后的索引
    order = scores.argsort()[::-1]
    # 存储保留的边界框的索引
    keep = []
    # 循环处理每个边界框
    while order.size > 0:
        # 取出当前最高得分的边界框的索引
        i = order[0]
        # 将其加入保留列表
        keep.append(i)
        # 计算当前边界框与其他剩余边界框的交集坐标
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        # 计算交集的宽度和高度，如果不存在交集则为0
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        # 计算交集的面积
        inter = w * h
        # 计算交并比
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 删除交并比大于阈值的边界框索引
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    # 返回保留的边界框索引
    return keep
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False):
    bs = prediction.shape[0]  # batch size
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000
    redundant = True  # require redundant detections
    merge = False  # use merge-NMS
    output = [np.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        # conf, j = x[:, 5:].max(1, keepdims=True)
        conf, j = x[:, 5:].max(1, keepdims=True), x[:, 5:].argmax(1, keepdims=True)
        x = np.concatenate((box, conf, j.astype(float)), axis=1)[conf.flatten() > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = v5nms(boxes, scores, iou_thres)  # NMS

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.matmul(weights, x[:, :4]).astype(float) / weights.sum(1, keepdims=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
    return output
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2




class YOLOX_ONNX:
    def __init__(self, model, zhixingdu=0.25, nmszhi=0.25):
        self.model = model # 模型
        self.input_shape = '416, 416'
        self.Yv = ''
        self.zhixingdu = zhixingdu
        self.nmszhi = nmszhi
        self.sffp16 = None

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.enable_mem_reuse = False
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2

        self.session = onnxruntime.InferenceSession(self.model, providers=['DmlExecutionProvider', 'CPUExecutionProvider'],
                                                        sess_options=sess_options)
        # 获取当前设备的ID
        device_id = onnxruntime.get_device()
        logger.info(f'推理类型-ONNX-{device_id}')

    def main(self, img1, xianshi=True):
        if self.Yv == 'YOLOX':
            try:
                img, ratio = preproc(img1, self.input_shape)
            except:
                self.input_shape = tuple(map(int, self.input_shape.split(',')))
                img, ratio = preproc(img1, self.input_shape)
            ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
            output = self.session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0], self.input_shape)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio

            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nmszhi, score_thr=self.zhixingdu)

            if xianshi == True:
                if dets is not None:
                    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                    vis(img1, final_boxes, final_scores, final_cls_inds, conf=self.zhixingdu)
            return dets
        elif self.Yv == 'YOLOV5':

            try:
                img = letterbox(img1, (self.input_shape[0], self.input_shape[1]), stride=32, auto=False)
                img = np.transpose(img, (2, 0, 1))
                if self.sffp16 == True:
                    im = np.float16(img)
                else:
                    im = np.float32(img)

                im /= 255.0  # 0 - 255 to 0.0 - 1.0 # 把张量归一化到0-1之间，注意要用浮点数除法
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                y = self.session.run(None, {self.session.get_inputs()[0].name: im})[0]

                pred = non_max_suppression(y, conf_thres=self.zhixingdu, iou_thres=self.nmszhi, agnostic=False)

                for i, det in enumerate(pred):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img1.shape).round()
                if xianshi == True:
                    # initialize annotator
                    if pred is not None:
                        final_boxes, final_scores, final_cls_inds = det[:, :4], det[:, 4], det[:, 5]
                        vis(img1, final_boxes, final_scores, final_cls_inds, conf=self.zhixingdu)

                return det

            except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument as e:
                print(e)
                # 使用正则表达式来捕获index: 2和index: 3后面的Expected的值
                pattern1 = r"index: [23] Got: \d+ Expected: (\d+)"
                # 使用re.findall方法来返回所有匹配的结果，每个结果是一个字符串，包含Expected的值
                e = str(e)
                match = re.findall(pattern1, e)

                pattern = r"expected: \(tensor\(float\)\)"  # 使用反斜杠转义括号
                string = "expected: (tensor(float))"
                match2 = re.search(pattern, string)
                if match:
                    expected_size_1, expected_size_2 = match[0], match[1]  # convert the matched string to integer
                    self.input_shape = f"{expected_size_1}, {expected_size_2}"
                    self.input_shape = tuple(map(int, self.input_shape.split(',')))
                    logger.error(f'已修改当前模型尺寸为：{self.input_shape}')
                elif match2:
                    self.sffp16 = False
                    logger.error(f'已修改当前模型精度为：{self.sffp16}')
                else:
                    logger.error(f'错误错误: {str(e)}')
        else:
            try:
                try:
                    img, ratio = preproc(img1, self.input_shape)
                except:
                    self.input_shape = tuple(map(int, self.input_shape.split(',')))
                    img, ratio = preproc(img1, self.input_shape)

                ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
                output = self.session.run(None, ort_inputs)
                predictions = demo_postprocess(output[0], self.input_shape)[0]

                boxes = predictions[:, :4]
                scores = predictions[:, 4:5] * predictions[:, 5:]

                boxes_xyxy = np.ones_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
                boxes_xyxy /= ratio
                dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nmszhi, score_thr=self.zhixingdu)
                self.Yv = 'YOLOX'
                if xianshi == True:
                    if dets is not None:
                        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                        vis(img1, final_boxes, final_scores, final_cls_inds, conf=self.zhixingdu)
                return dets
            except:
                try:
                    try:
                        img = letterbox(img1, (self.input_shape[0], self.input_shape[1]), stride=32,
                                        auto=False)  # only pt use auto=True, but we are onnx
                    except:
                        self.input_shape = tuple(map(int, self.input_shape.split(',')))
                        img = letterbox(img1, (self.input_shape[0], self.input_shape[1]), stride=32,
                                        auto=False)  # only pt use auto=True, but we are onnx
                    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    img = np.ascontiguousarray(img)
                    img1 = np.ascontiguousarray(img1)
                    im = img.astype(np.float32)  # convert to float32

                    if self.sffp16 == True:
                        im = im.astype(np.float16)  # 把张量转换为float16类型
                    im /= 255.0  # 0 - 255 to 0.0 - 1.0 # 把张量归一化到0-1之间，注意要用浮点数除法
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim


                    y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]


                    pred = non_max_suppression(y, conf_thres=self.zhixingdu, iou_thres=self.nmszhi, agnostic=False)

                    for i, det in enumerate(pred):
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img1.shape).round()
                    self.Yv = 'YOLOV5'
                    if xianshi == True:
                        # initialize annotator
                        if pred is not None:
                            final_boxes, final_scores, final_cls_inds = det[:, :4], det[:, 4], det[:, 5]
                            vis(img1, final_boxes, final_scores, final_cls_inds, conf=self.zhixingdu)
                    return det
                except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument as e:
                    # 使用正则表达式来捕获index: 2和index: 3后面的Expected的值
                    pattern1 = r"index: [23] Got: \d+ Expected: (\d+)"
                    # 使用re.findall方法来返回所有匹配的结果，每个结果是一个字符串，包含Expected的值
                    e = str(e)
                    match = re.findall(pattern1, e)
                    pattern = r"expected: \(tensor\(float\)\)"  # 使用反斜杠转义括号
                    string = "expected: (tensor(float))"
                    match2 = re.search(pattern, string)
                    if match:
                        expected_size_1, expected_size_2 = match[0], match[1]  # convert the matched string to integer
                        self.input_shape = f"{expected_size_1}, {expected_size_2}"
                        self.input_shape = tuple(map(int, self.input_shape.split(',')))
                        logger.error(f'已修改当前模型尺寸为：{self.input_shape}')
                    elif match2:
                        self.sffp16 = False
                        logger.error(f'已修改当前模型精度为：{self.sffp16}')
                    else:
                        logger.error(f'错误错误: {str(e)}')



# -------------------------------------------定义一个用于推理的类-------------------------------------------




# -------------------------------------------定义一个用于盒子的类-------------------------------------------

kmbox = serial.Serial()


def kmbox_is_open() -> bool:
    return kmbox.isOpen()


def init_kmbox(port: int, baud: int) -> bool:
    try:
        kmbox.port = f"COM{port}"
        kmbox.baudrate = baud
        kmbox.open()

        if kmbox.isOpen():
            kmbox.write("import km\r\n".encode())
            return True
        else:
            return False
    except:
        return False

def kmbox_moveR(x: int, y: int):
    if kmbox.isOpen():
        kmbox.write(f"km.move({x},{y})\r\n".encode())

def close_kmbox():
    if kmbox.isOpen():
        kmbox.close()

# ser=serial.Serial('COM3', 120000) #端口
# ser.write('import km\r'.encode('utf-8'))
#
# def mouse_xy(x, y):
#     operation = 'km.move(' + str(x) + ', ' + str(y) + ')\r'
#     ser.write(operation.encode('utf-8'))
#
# mouse_xy(100,0)


# -------------------------------------------定义一个用于盒子的类-------------------------------------------


