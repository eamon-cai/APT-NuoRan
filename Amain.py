import cv2
import math
import dxshot
import ctypes
import configparser
import win32gui
import winsound
import time
import numpy as np
from pynput import mouse, keyboard
from win32gui import FindWindow, SetWindowPos
from win32con import HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE

from mss import mss
from Toolset import HumanClicker
from Toolset import YOLOX_ONNX
from Toolset import ctypes_moveR
from Toolset import mouse_move
from Toolset import Capture
from Toolset import PID, ADRC
from Toolset import kalmanP

from loguru import logger
from ctypes import windll



windll.winmm.timeBeginPeriod(1)
stop = windll.kernel32.Sleep
button_mzr = False
button_mzl = False
button_kzq = False
button_kzy = False
button_kzz = False

xianshi = False
tuichu = False
suoding = True
toushen = True
qidong = False
qiehuan = 0

mss = []

kd = False
kd_1 = False
kd_2 = False
kd_3 = False
kd_4 = False
kd_5 = False




def on_press(key):
    global button_kzz, button_kzy
    """定义按下时候的响应，参数传入key"""
    try:
        if str(key) == text12:
            button_kzz = True
        elif str(key) == text13:
            button_kzy = True
    except:
        pass

def on_release(key):
    global tuichu, xianshi, suoding, toushen, button_kzq, button_kzy, button_kzz, qiehuan
    try:
        if str(key) == text1:
            tuichu = True
            stop(50)
            tuichu = False
            listener1.stop()
            listener2.stop()
            winsound.Beep(400, 200)
        elif str(key) == text2:
            xianshi = not xianshi
            if xianshi:
                logger.debug('开启显示')
            else:
                logger.debug('关闭显示')
        elif str(key) == text3:
            suoding = not suoding
            if suoding:
                winsound.Beep(800, 200)
                logger.debug('开启锁定')
            else:
                winsound.Beep(400, 200)
                logger.debug('关闭锁定')
        elif str(key) == text4:
            toushen = not toushen
            if toushen:
                logger.debug('身体')
            else:
                logger.debug('头')
        elif str(key) == text12:
            button_kzz = False
        elif str(key) == text13:
            button_kzy = False
        elif str(key) == text14:
            button_kzq = not button_kzq
            if button_kzq:
                winsound.Beep(800, 200)
            else:
                winsound.Beep(400, 200)
        elif str(key) == text15:
            qiehuan = 0
            winsound.Beep(800, 200)
            logger.debug('激活①②标签')
        elif str(key) == text16:
            qiehuan = 1
            winsound.Beep(600, 200)
            logger.debug('激活①标签')
        elif str(key) == text17:
            qiehuan = 2
            winsound.Beep(400, 200)
            logger.debug('激活②标签')
    except:
        pass

def listen_key_nblock():
    global listener1
    listener1 = keyboard.Listener(
        on_press=on_press, on_release=on_release
    )
    listener1.start()

def on_click(x, y, button, pressed):
    global button_mzl, button_mzr, button_kzq, xianshi, suoding, toushen, qiehuan
    try:
        if text12 == str(button):
            if pressed == True:
                button_mzl = True
            else:
                button_mzl = False
        elif text13 == str(button):
            if pressed == True:
                button_mzr = True
            else:
                button_mzr = False
        elif text14 == str(button):
            if pressed == True:
                button_kzq = not button_kzq
                if button_kzq:
                    winsound.Beep(800, 200)
                else:
                    winsound.Beep(400, 200)
        elif text15 == str(button):
            if pressed == True:
                qiehuan = 0
                winsound.Beep(800, 200)
                logger.debug('激活①②标签')
        elif text16 == str(button):
            if pressed == True:
                qiehuan = 1
                winsound.Beep(600, 200)
                logger.debug('激活①标签')
        elif text17 == str(button):
            if pressed == True:
                qiehuan = 2
                winsound.Beep(400, 200)
                logger.debug('激活②标签')
        elif text2 == str(button):
            if pressed == True:
                xianshi = not xianshi
                if xianshi:
                    logger.debug('开启显示')
                else:
                    logger.debug('关闭显示')
        elif text3 == str(button):
            if pressed == True:
                suoding = not suoding
                if suoding:
                    winsound.Beep(800, 200)
                    logger.debug('开启锁定')
                else:
                    winsound.Beep(400, 200)
                    logger.debug('关闭锁定')
        elif text4 == str(button):
            if pressed == True:
                toushen = not toushen
                if toushen:
                    logger.debug('身体')
                else:
                    logger.debug('头')

    except:
        pass

def listen_mouse_nblock():
    global listener2
    listener2 = mouse.Listener(
        on_click=on_click,
    )
    listener2.start()


def jieshu():
    global tuichu, qidong
    if qidong == True:
        tuichu = True
        listener1.stop()
        listener2.stop()
        stop(50)
        tuichu = False
        winsound.Beep(400, 200)
        qidong = False
    else:
        try:
            listener1.stop()
            listener2.stop()
        except:
            pass

def main():
    global text12, text13, qidong, text1, text2, text3, text4, text14, save_jx, text15, tuichu
    global button_mzl, button_mzr, button_kzq, button_kzy, button_kzz, button_kzq, text16, text17
    winsound.Beep(800, 200)

    import os
    current_path = os.path.dirname(os.path.abspath(__file__))
    config = configparser.ConfigParser()
    # 从config.ini文件中读取参数和值
    config.read("config.ini")

    try:
        model = config["Universal"]["Model"]
        msing_range = int(config["ms"]["msing_Range"])

        mobile_type = config["Universal"]["Mobile_Type"]
        screenshot_type = config["Universal"]["Screenshot_Type"]

        p_x = float(config["Tpid_x"]["p"])
        i_x = float(config["Tpid_x"]["i"])
        d_x = float(config["Tpid_x"]["d"])
        c_x = float(config["Tpid_x"]["c"])

        p_y = float(config["Tpid_y"]["p"])
        i_y = float(config["Tpid_y"]["i"])
        d_y = float(config["Tpid_y"]["d"])
        c_y = float(config["Tpid_y"]["c"])

        text1 = config["vk"]["Eixt"]
        text2 = config["vk"]["Show"]
        text3 = config["vk"]["Target"]
        text4 = config["vk"]["HB"]
        text12 = config["vk"]["Lock_1"]
        text13 = config["vk"]["Lock_2"]
        text14 = config["vk"]["Lock_3"]
        text15 = config["vk"]["Label_12"]
        text16 = config["vk"]["Label_1"]
        text17 = config["vk"]["Label_2"]

        head_msing_ratio = float(config["Universal"]["head_msing_ratio"])
        body_msing_ratio = float(config["Universal"]["body_msing_ratio"])
        scr = float(config["Universal"]["scr"])
        nms = float(config["Universal"]["nms"])
    except KeyError as e:
        logger.error(f"未知错误, {e}")
        return

    logger.info('获取配置参数---完成')

    # 获取 user32.dll 的句柄
    user32 = ctypes.windll.user32

    # 获取主要显示器的分辨率
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)

    imgsize = int(320)  # 只能用320
    offset = int(imgsize / 2)
    fbl = [int(width), int(height)]

    szb = [int((fbl[0] / 2) - (imgsize / 2)), int((fbl[1] / 2) - (imgsize / 2))]
    xzb = [int((fbl[0] / 2) + (imgsize / 2)), int((fbl[1] / 2) + (imgsize / 2))]

    region = (szb[0], szb[1], xzb[0], xzb[1])
    logger.info('获取分辨率参数---完成')


    model_onnx = f"{current_path}/weights/{model}"
    logger.info('模型参数设置---完成')


    cam = dxshot.create(output_color="BGR")
    logger.info('截图类型---DX截图')

    Image_name = 'onnx'
    csh = YOLOX_ONNX(model=model_onnx, zhixingdu=scr, nmszhi=nms)

    tspidx = PID(dim=1, Kp=p_x, Ki=i_x, Kd=d_x)
    tspidy = PID(dim=1, Kp=p_y, Ki=i_y, Kd=d_y)
    adrc_x = ADRC(w0=0.00001, b0=c_x, w_n=1, sigma=0.00001, time_delta=0.00001)
    adrc_y = ADRC(w0=0.00001, b0=c_y, w_n=1, sigma=0.00001, time_delta=0.00001)
    logger.info('移动类型---PID移动模式')

    listen_key_nblock()
    listen_mouse_nblock()
    logger.info('线程启动---完成')




    if screenshot_type == 'DX截图':
        cam = dxshot.create(output_color="BGR")
        logger.info('截图类型---DX截图')
    elif screenshot_type == 'Mss截图':
        msCP = {"top": szb[1], "left": szb[0], "width": imgsize, "height": imgsize}
        logger.info('截图类型---Mss截图')
    elif screenshot_type == '句柄截图':
        stop(2000)
        # 创建一个Capture类的实例
        Cp = Capture()
        Cp.Init(win32gui.GetForegroundWindow(), imgsize, imgsize, 0, 0)
        logger.info('截图类型---句柄截图')
    elif screenshot_type == '桌面截图':
        # 创建一个Capture类的实例
        Cp = Capture()
        Cp.Init(win32gui.GetDesktopWindow(), imgsize, imgsize, 0, 0)
        logger.info('截图类型---桌面截图')
    else:
        logger.error('截图类型-未选择！！！')
        return


    dist_list = []
    boxs = []
    target_id = None
    r_move = HumanClicker()

    klm = kalmanP(1, 3)

    qidong = True
    last_save_time1 = time.time()
    logger.warning('-------------------------------开始循环-------------------------------')
    while True:
        t0 = time.time()
        if screenshot_type == 'DX截图':
            try:
                img = cam.grab(region=region)
            except:
                winsound.Beep(400, 200)
                tuichu = True
                cv2.destroyAllWindows()
                jieshu()
                logger.error('截图报错，请重新启动')
                break
        elif screenshot_type == '桌面截图' or screenshot_type == '句柄截图':
            img = Cp.capture()
        elif screenshot_type == 'Mss截图':
            with mss() as sct:
                img = sct.grab(msCP)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = Cp.capture()
        try:
            # 从config.ini文件中读取参数和值
            config.read("config.ini")

            text1 = config["vk"]["Eixt"]
            text2 = config["vk"]["Show"]
            text3 = config["vk"]["Target"]
            text4 = config["vk"]["HB"]
            text12 = config["vk"]["Lock_1"]
            text13 = config["vk"]["Lock_2"]
            text14 = config["vk"]["Lock_3"]
            text15 = config["vk"]["Label_12"]
            text16 = config["vk"]["Label_1"]
            text17 = config["vk"]["Label_2"]

            p_x = float(config["Tpid_x"]["p"])
            i_x = float(config["Tpid_x"]["i"])
            d_x = float(config["Tpid_x"]["d"])
            c_x = float(config["Tpid_x"]["c"])

            p_y = float(config["Tpid_y"]["p"])
            i_y = float(config["Tpid_y"]["i"])
            d_y = float(config["Tpid_y"]["d"])
            c_y = float(config["Tpid_y"]["c"])
            msing_range = int(config["ms"]["msing_Range"])

            head_msing_ratio = float(config["Universal"]["head_msing_ratio"])
            body_msing_ratio = float(config["Universal"]["body_msing_ratio"])

            tspidx.gengxing(Kp=p_x, Ki=i_x, Kd=d_x)
            tspidy.gengxing(Kp=p_y, Ki=i_y, Kd=d_y)
            adrc_x.gengxing(c_x)
            adrc_y.gengxing(c_y)
        except:
            logger.error('实时调参错误！！！')

        if img is not None:
            if xianshi is True:
                jc = csh.main(img)
            else:
                jc = csh.main(img, False)
            mss.clear()
            boxs.clear()
            dist_list.clear()

            if jc is not None:
                final_boxes, final_scores, final_cls_inds = jc[:, :4], jc[:, 4], jc[:, 5]
            else:
                final_boxes, final_scores, final_cls_inds = [], [], []

            for i in range(len(final_boxes)):  # 用一个循环遍历所有检测到的物体
                box = final_boxes[i]  # 获取第i个物体的边框坐标
                cls = int(final_cls_inds[i])  # 获取类型
                line2 = [cls, int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                boxs.append(line2)

            # # -----------------------------------------------可视化-----------------------------------
            boxs_pd = klm.predict(boxs)
            for box in boxs_pd:
                if box is None:
                    continue
                cls = box[0]
                x0 = box[1]
                y0 = box[2]
                x1 = box[3]
                y1 = box[4]
                id = box[5]

                cx = x1 - ((x1 - x0) // 2)
                cy = y1 - ((y1 - y0) // 2)
                height = y0 - y1

                line3 = [int(cls), int(cx), int(cy), int(height), int(id)]
                mss.append(line3)
            mss_copy = mss

            if qiehuan == 0:
                mss_copy = [x for x in mss_copy if x[0] in [0, 1]]
            elif qiehuan == 1:
                mss_copy = [x for x in mss_copy if x[0] in [0]]
                if 0 in [x[0] for x in mss_copy]:
                    mss_copy = [x for x in mss_copy if x[0] in [0]]
            elif qiehuan == 2:
                mss_copy = [x for x in mss_copy if x[0] in [1]]
                if 1 in [x[0] for x in mss_copy]:
                    mss_copy = [x for x in mss_copy if x[0] in [1]]
            else:
                mss_copy = [x for x in mss_copy if x[0] in [0, 1]]

            if len(mss_copy):

                found = False
                for ms in mss_copy:  # 遍历mss_copy中的每一个元素
                    if ms[-1] is target_id:  # 如果ms的最后一个元素等于target_id
                        tag, x, y, height, id = ms
                        found = True
                        break  # 找到了就跳出循环
                if not found:  # 如果没有找到当前锁定的目标
                    dist_list = []
                    for _, x_c, y_c, _, _ in mss_copy:
                        dist = math.sqrt((x_c - offset) ** 2 + (y_c - offset) ** 2)
                        dist_list.append(dist)

                    min_dist = min(dist_list)
                    nearest_index = dist_list.index(min_dist)
                    tag, x, y, height, id = mss_copy[nearest_index]

                    # 更新目标ID
                    target_id = id


                if (x - offset) ** 2 + (y - offset) ** 2 < msing_range ** 2:
                    y = y - (height / 2)

                    if toushen is True:
                        y = y + (height * body_msing_ratio)
                    else:
                        y = y + (height * head_msing_ratio)
                    cv2.line(img, (offset, offset), (int(x), int(y)), (255, 255, 255), 1)

                    x = int(offset - x)
                    y = int(offset - y)

                    move_x = int(tspidx.update(x))
                    move_y = int(tspidy.update(y))

                    move_x = -adrc_x.update(move_x, move_x)
                    move_y = -adrc_y.update(move_y, move_y)

                    if suoding is True:
                        if button_mzl is True or button_mzr is True or button_kzq is True or button_kzy is True or button_kzz is True:
                            path = r_move.move_to((int(move_x), int(move_y)))
                            last_point = None
                            for point in path:
                                last_point = point
                            pass
                        else:
                            PID.reset(tspidx)
                            PID.reset(tspidy)
                            ADRC.reset(adrc_x)
                            ADRC.reset(adrc_y)
                    else:
                        PID.reset(tspidx)
                        PID.reset(tspidy)
                        ADRC.reset(adrc_x)
                        ADRC.reset(adrc_y)
                else:
                    PID.reset(tspidx)
                    PID.reset(tspidy)
                    ADRC.reset(adrc_x)
                    ADRC.reset(adrc_y)
            if xianshi is True:
                cv2.circle(img, (offset, offset), msing_range, (255, 255, 0), 1)
                cv2.putText(img, f'FPS: {1000 / ((time.time() - t0) * 1000):.0f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                cv2.namedWindow(Image_name, cv2.WINDOW_AUTOSIZE)
                cv2.circle(img, (offset, offset), 2, (0, 0, 255), -1)
                cv2.imshow(Image_name, img)
                SetWindowPos(FindWindow(None, Image_name), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
                cv2.waitKey(1)
            else:
                cv2.destroyAllWindows()
        if tuichu is True:
            cv2.destroyAllWindows()
            del cam
            del csh
            break
