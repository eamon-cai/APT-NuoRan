import os
import ctypes
import sys

import gradio as gr
import configparser
from loguru import logger
from Amain import main, jieshu

config = configparser.ConfigParser()
# 从config.ini文件中读取参数和值
config.read("config.ini")


class Getinitialization:
    def __init__(self):
        self.all_onnx = []
        self.keys = [
            '不启用',
            '鼠标左键',
            '鼠标中键',
            '鼠标右键',
            '上侧键',
            '下侧键',
            '左Alt',
            '右Alt',
            '左Ctrl',
            '右Ctrl',
            '退格键',
            '大写键',
            '向左键',
            '向右键',
            '向上键',
            '向下键',
            '回车键',
            '退出键',
            '删除键',
            '空格键',
            'End',
            'Home',
            'Shift',
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "Num0",
            "Num1",
            "Num2",
            "Num3",
            "Num4",
            "Num5",
            "Num6",
            "Num7",
            "Num8",
            "Num9",
            "Num0",
            'F1',
            'F2',
            'F3',
            'F4',
            'F5',
            'F6',
            'F7',
            'F8',
            'F9',
            'F10',
            'F11',
            'F12',
            'Insert',
            'Num_Lock',
            'Pg_Down',
            'PG_Up',
            'Tab',

        ]

    def Set(self):
        try:
            # 定义要检测的文件后缀
            suffix = ".onnx"

            # 定义当前文件夹路径
            current_path = os.path.dirname(os.path.abspath(__file__))

            # 定义weights文件夹的路径
            weights_path = os.path.join(current_path, "weights")

            # 定义一个空列表来存储所有文件的路径
            self.all_onnx = []

            # 遍历weights文件夹中的所有文件和子文件夹
            for root, dirs, files in os.walk(weights_path):
                # 遍历所有文件
                for file in files:
                    if file.endswith(suffix):
                        # 判断是否是子文件夹中的文件
                        if root != weights_path:
                            # 获取子文件夹的名称
                            subfolder_name = os.path.basename(root)
                            # 在文件路径后面加上括号和子文件夹的名称
                            file = file + " --(" + subfolder_name + ")"
                        self.all_onnx.append(file)
        except:
            self.all_onnx = []


Getinitialize = Getinitialization()


class Setinitialization:
    def __init__(self):
        self.model = None               # 模型
        self.mobile_type = None         # 移动类型
        self.screenshot_type = None     # 截图类型
        self.msing_range = None        # 范围

        # PID
        self.p_x = None
        self.i_x = None
        self.d_x = None
        self.c_x = None

        self.p_y = None
        self.i_y = None
        self.d_y = None
        self.c_y = None

        # 键位Key
        self.eixt_key = None
        self.show_key = None
        self.target_key = None
        self.hb_key = None
        self.key_lock_key_1 = None
        self.key_lock_key_2 = None
        self.key_lock_key_3 = None
        self.key_label_12 = None
        self.key_label_1 = None
        self.key_label_2 = None



        # 通用设置
        self.head_msing_ratio = None
        self.body_msing_ratio = None
        self.scr = None
        self.nms = None



    def Set(self):





        try:
            self.mobile_type = config["Universal"]["Mobile_Type"]
        except:
            self.mobile_type = ''

        try:
            self.screenshot_type = config["Universal"]["Screenshot_Type"]
        except:
            self.screenshot_type = ''

        try:
            self.model = config["Universal"]["Model"]
        except:
            self.model = ''

        try:
            self.msing_range = config["ms"]["msing_Range"]
        except:
            self.msing_range = 160

        try:
            self.adrc_start = config["Universal"]["adrc_start"]
        except:
            self.adrc_start = ''

        try:
            self.p_x = config["Tpid_x"]["p"]
        except:
            self.p_x = 0

        try:
            self.i_x = config["Tpid_x"]["i"]
        except:
            self.i_x = 0

        try:
            self.d_x = config["Tpid_x"]["d"]
        except:
            self.d_x = 0

        try:
            self.c_x = config["Tpid_x"]["c"]
        except:
            self.c_x = 0

        try:
            self.p_y = config["Tpid_y"]["p"]
        except:
            self.p_y = 0

        try:
            self.i_y = config["Tpid_y"]["i"]
        except:
            self.i_y = 0

        try:
            self.d_y = config["Tpid_y"]["d"]
        except:
            self.d_y = 0

        try:
            self.c_y = config["Tpid_y"]["c"]
        except:
            self.c_y = 0

        try:
            self.eixt_key = KeyG(config["vk"]["Eixt"])
        except:
            self.eixt_key = ''

        try:
            self.show_key = KeyG(config["vk"]["Show"])
        except:
            self.show_key = ''

        try:
            self.target_key = KeyG(config["vk"]["Target"])
        except:
            self.target_key = ''

        try:
            self.hb_key = KeyG(config["vk"]["HB"])
        except:
            self.hb_key = ''

        try:
            self.key_lock_key_1 = KeyG(config["vk"]["Lock_1"])
        except:
            self.key_lock_key_1 = ''

        try:
            self.key_lock_key_2 = KeyG(config["vk"]["Lock_2"])
        except:
            self.key_lock_key_2 = ''

        try:
            self.key_lock_key_3 = KeyG(config["vk"]["Lock_3"])
        except:
            self.key_lock_key_3 = ''

        try:
            self.key_label_12 = None
            self.key_label_12 = KeyG(config["vk"]["Label_12"])
        except:
            self.key_label_12 = ''

        try:
            self.key_label_1 = None
            self.key_label_1 = KeyG(config["vk"]["Label_1"])
        except:
            self.key_label_1 = ''

        try:
            self.key_label_2 = None
            self.key_label_2 = KeyG(config["vk"]["Label_2"])
        except:
            self.key_label_2 = ''

        try:
            self.head_msing_ratio = config["Universal"]["head_msing_ratio"]
        except:
            self.head_msing_ratio = 0.9

        try:
            self.body_msing_ratio = config["Universal"]["body_msing_ratio"]
        except:
            self.body_msing_ratio = 0.65

        try:
            self.scr = config["Universal"]["scr"]
        except:
            self.scr = 0.6

        try:
            self.nms = config["Universal"]["nms"]
        except:
            self.nms = 0.25



Setinitialize = Setinitialization()


def KeyG(text):
    if text == "'a'":
        return 'A'
    elif text == "'b'":
        return 'B'
    elif text == "'c'":
        return 'C'
    elif text == "'d'":
        return 'D'
    elif text == "'e'":
        return 'E'
    elif text == "'f'":
        return 'F'
    elif text == "'g'":
        return 'G'
    elif text == "'h'":
        return 'H'
    elif text == "'i'":
        return 'I'
    elif text == "'j'":
        return 'J'
    elif text == "'k'":
        return 'K'
    elif text == "'l'":
        return 'L'
    elif text == "'m'":
        return 'M'
    elif text == "'n'":
        return 'N'
    elif text == "'o'":
        return 'O'
    elif text == "'p'":
        return 'P'
    elif text == "'q'":
        return 'Q'
    elif text == "'r'":
        return 'R'
    elif text == "'s'":
        return 'S'
    elif text == "'t'":
        return 'T'
    elif text == "'u'":
        return 'U'
    elif text == "'v'":
        return 'V'
    elif text == "'w'":
        return 'W'
    elif text == "'x'":
        return 'X'
    elif text == "'y'":
        return 'Y'
    elif text == "'z'":
        return 'Z'
    elif text == "'1'":
        return '1'
    elif text == "'2'":
        return '2'
    elif text == "'3'":
        return '3'
    elif text == "'4'":
        return '4'
    elif text == "'5'":
        return '5'
    elif text == "'6'":
        return '6'
    elif text == "'7'":
        return '7'
    elif text == "'8'":
        return '8'
    elif text == "'9'":
        return '9'
    elif text == "'0'":
        return '0'
    elif text == "<96>":
        return 'Num0'
    elif text == "<97>":
        return 'Num1'
    elif text == "<98>":
        return 'Num2'
    elif text == "<99>":
        return 'Num3'
    elif text == "<100>":
        return 'Num4'
    elif text == "<101>":
        return 'Num5'
    elif text == "<102>":
        return 'Num6'
    elif text == "<103>":
        return 'Num7'
    elif text == "<104>":
        return 'Num8'
    elif text == "<105>":
        return 'Num9'
    elif text == 'Key.alt_l':
        return '左Alt'
    elif text == 'Key.alt_r':
        return '右Alt'
    elif text == 'Key.caps_lock':
        return '大写键'
    elif text == 'Key.backspace':
        return '退格键'
    elif text == 'Key.ctrl_l':
        return '左Ctrl'
    elif text == 'Key.ctrl_r':
        return '右Ctrl'
    elif text == 'Key.shift':
        return 'Shift'
    elif text == 'Key.space':
        return '空格键'
    elif text == 'Key.delete':
        return '删除键'
    elif text == 'Key.down':
        return '向下键'
    elif text == 'Key.up':
        return '向上键'
    elif text == 'Key.left':
        return '向左键'
    elif text == 'Key.right':
        return '向右键'
    elif text == 'Key.end':
        return 'End'
    elif text == 'Key.enter':
        return '回车键'
    elif text == 'Key.esc':
        return '退出键'
    elif text == 'Key.f1':
        return 'F1'
    elif text == 'Key.f2':
        return 'F2'
    elif text == 'Key.f3':
        return 'F3'
    elif text == 'Key.f4':
        return 'F4'
    elif text == 'Key.f5':
        return 'F5'
    elif text == 'Key.f6':
        return 'F6'
    elif text == 'Key.f7':
        return 'F7'
    elif text == 'Key.f8':
        return 'F8'
    elif text == 'Key.f9':
        return 'F9'
    elif text == 'Key.f10':
        return 'F10'
    elif text == 'Key.f11':
        return 'F11'
    elif text == 'Key.f12':
        return 'F12'
    elif text == 'Key.home':
        return 'Home'
    elif text == 'Key.insert':
        return 'Insert'
    elif text == 'Key.num_lock':
        return 'Num_Lock'
    elif text == 'Key.page_down':
        return 'Pg_Down'
    elif text == 'Key.page_up':
        return 'Pg_Up'
    elif text == 'Key.tab':
        return 'Tab'
    elif text == 'Button.left':
        return '鼠标左键'
    elif text == 'Button.middle':
        return '鼠标中键'
    elif text == 'Button.right':
        return '鼠标右键'
    elif text == 'Button.x2':
        return '上侧键'
    elif text == 'Button.x1':
        return '下侧键'

def KeyS(text):
    if text == 'A':
        return "'a'"
    elif text == 'B':
        return "'b'"
    elif text == 'C':
        return "'c'"
    elif text == 'D':
        return "'d'"
    elif text == 'E':
        return "'e'"
    elif text == 'F':
        return "'f'"
    elif text == 'G':
        return "'g'"
    elif text == 'H':
        return "'h'"
    elif text == 'I':
        return "'i'"
    elif text == 'J':
        return "'j'"
    elif text == 'K':
        return "'k'"
    elif text == 'L':
        return "'l'"
    elif text == 'M':
        return "'m'"
    elif text == 'N':
        return "'n'"
    elif text == 'O':
        return "'o'"
    elif text == 'P':
        return "'p'"
    elif text == 'Q':
        return "'q'"
    elif text == 'R':
        return "'r'"
    elif text == 'S':
        return "'s'"
    elif text == 'T':
        return "'t'"
    elif text == 'U':
        return "'u'"
    elif text == 'V':
        return "'v'"
    elif text == 'W':
        return "'w'"
    elif text == 'X':
        return "'x'"
    elif text == 'Y':
        return "'y'"
    elif text == 'Z':
        return "'z'"
    elif text == '1':
        return "'1'"
    elif text == '2':
        return "'2'"
    elif text == '3':
        return "'3'"
    elif text == '4':
        return "'4'"
    elif text == '5':
        return "'5'"
    elif text == '6':
        return "'6'"
    elif text == '7':
        return "'7'"
    elif text == '8':
        return "'8'"
    elif text == '9':
        return "'9'"
    elif text == '0':
        return "'0'"
    elif text == 'Num0':
        return "<96>"
    elif text == 'Num1':
        return "<97>"
    elif text == 'Num2':
        return "<98>"
    elif text == 'Num3':
        return "<99>"
    elif text == 'Num4':
        return "<100>"
    elif text == 'Num5':
        return "<101>"
    elif text == 'Num6':
        return "<102>"
    elif text == 'Num7':
        return "<103>"
    elif text == 'Num8':
        return "<104>"
    elif text == 'Num9':
        return "<105>"
    elif text == '左Alt':
        return 'Key.alt_l'
    elif text == '右Alt':
        return 'Key.alt_r'
    elif text == '大写键':
        return 'Key.caps_lock'
    elif text == '退格键':
        return 'Key.backspace'
    elif text == '左Ctrl':
        return 'Key.ctrl_l'
    elif text == '右Ctrl':
        return 'Key.ctrl_r'
    elif text == 'Shift':
        return 'Key.shift'
    elif text == '空格键':
        return 'Key.space'
    elif text == '删除键':
        return 'Key.delete'
    elif text == '向下键':
        return 'Key.down'
    elif text == '向上键':
        return 'Key.up'
    elif text == '向左键':
        return 'Key.left'
    elif text == '向右键':
        return 'Key.right'
    elif text == 'End':
        return 'Key.end'
    elif text == '回车键':
        return 'Key.enter'
    elif text == '退出键':
        return 'Key.esc'
    elif text == 'F1':
        return 'Key.f1'
    elif text == 'F2':
        return 'Key.f2'
    elif text == 'F3':
        return 'Key.f3'
    elif text == 'F4':
        return 'Key.f4'
    elif text == 'F5':
        return 'Key.f5'
    elif text == 'F6':
        return 'Key.f6'
    elif text == 'F7':
        return 'Key.f7'
    elif text == 'F8':
        return 'Key.f8'
    elif text == 'F9':
        return 'Key.f9'
    elif text == 'F10':
        return 'Key.f10'
    elif text == 'F11':
        return 'Key.f11'
    elif text == 'F12':
        return 'Key.f12'
    elif text == 'Home':
        return 'Key.home'
    elif text == 'Insert':
        return 'Key.insert'
    elif text == 'Num_Lock':
        return 'Key.num_lock'
    elif text == 'Pg_Down':
        return 'Key.page_down'
    elif text == 'Pg_Up':
        return 'Key.page_up'
    elif text == 'Tab':
        return 'Key.tab'
    elif text == '鼠标左键':
        return 'Button.left'
    elif text == '鼠标中键':
        return 'Button.middle'
    elif text == '鼠标右键':
        return 'Button.right'
    elif text == '上侧键':
        return 'Button.x2'
    elif text == '下侧键':
        return 'Button.x1'


def save(model, mobile_type, screenshot_type, head_msing_ratio, body_msing_ratio, scr, nms, msing_range,
         p_x, i_x, d_x, c_x, p_y, i_y, d_y, c_y,
         eixt_key, show_key, target_key, hb_key, key_lock_key_1, key_lock_key_2,
         key_lock_key_3, key_label_12, key_label_1, key_label_2):



    config["Universal"] = {}
    config["Universal"]["Model"] = str(model)
    config["Universal"]["Mobile_Type"] = str(mobile_type)
    config["Universal"]["Screenshot_Type"] = str(screenshot_type)
    config["Universal"]["head_msing_ratio"] = str(head_msing_ratio)
    config["Universal"]["body_msing_ratio"] = str(body_msing_ratio)
    config["Universal"]["scr"] = str(scr)
    config["Universal"]["nms"] = str(nms)


    config["Tpid_x"] = {}
    config["Tpid_x"]["p"] = str(p_x)
    config["Tpid_x"]["i"] = str(i_x)
    config["Tpid_x"]["d"] = str(d_x)
    config["Tpid_x"]["c"] = str(c_x)

    config["Tpid_y"] = {}
    config["Tpid_y"]["p"] = str(p_y)
    config["Tpid_y"]["i"] = str(i_y)
    config["Tpid_y"]["d"] = str(d_y)
    config["Tpid_y"]["c"] = str(c_y)

    config["ms"] = {}
    config["ms"]["msing_Range"] = str(msing_range)


    config["vk"] = {}
    config["vk"]["Eixt"] = str(KeyS(eixt_key))
    config["vk"]["Show"] = str(KeyS(show_key))
    config["vk"]["Target"] = str(KeyS(target_key))
    config["vk"]["HB"] = str(KeyS(hb_key))

    config["vk"]["Lock_1"] = str(KeyS(key_lock_key_1))
    config["vk"]["Lock_2"] = str(KeyS(key_lock_key_2))
    config["vk"]["Lock_3"] = str(KeyS(key_lock_key_3))
    config["vk"]["Label_12"] = str(KeyS(key_label_12))
    config["vk"]["Label_1"] = str(KeyS(key_label_1))
    config["vk"]["Label_2"] = str(KeyS(key_label_2))

    # 将参数和值写入config.ini文件中
    with open("config.ini", "w") as f:
        config.write(f)




# 定义一个函数，接受一个文件夹路径作为参数
def open_folder():
    # 定义当前文件夹路径
    current_path = os.path.dirname(os.path.abspath(__file__))

    # 定义weights文件夹的路径
    weights_path = os.path.join(current_path, "weights")

    # 判断文件夹路径是否存在
    if os.path.exists(weights_path):
        # 使用os.startfile()方法打开文件夹
        os.startfile(weights_path)
    else:
        # 如果文件夹路径不存在，打印错误信息
        logger.error("无weights文件夹！！")

# 定义一个函数，检查当前用户是否是管理员
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def restart_program():
    # 用当前的Python解释器来执行一个新的程序，并替换当前的进
    os.execl(sys.executable, sys.executable, *sys.argv)

def uimain():


    logger.info('获取当前是否拥有管理员权限...')
    # 如果不是管理员，就执行你的程序代码
    if not is_admin():
        logger.error('无管理员权限')
        logger.debug('即将重启软件...获取管理员权限')
        # 如果不是管理员，就用runas命令重新运行程序
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        # 关闭原来的进程
        sys.exit(0)
    else:
        logger.info('当前已拥有管理员权限...')
        Getinitialize.Set()
        Setinitialize.Set()
        logger.info('当前已初始化成功...')
        logger.info('启动软件中...')
    with gr.Blocks() as demo:
        with gr.Tab("通用设置"):
            with gr.Row(): #并行显示，可开多列
                with gr.Column(): # 并列显示，可开多行
                    with gr.Column():  # 并列显示，可开多行
                        model = gr.Dropdown(choices=Getinitialize.all_onnx, value=Setinitialize.model, label="ONNX设置",
                                             info="选择要使用的模型")  # 单选
                    with gr.Row():  # 并行显示，可开多列
                        mobile_type = gr.Radio(["罗技移动", "Win移动"],
                                                 label="移动类型", info="选择要使用的移动类型",
                                                 value=Setinitialize.mobile_type)  # 单选

                        screenshot_type = gr.Radio(["DX截图", "句柄截图", "桌面截图", "Mss截图"],
                                                label="截图类型", info="选择要使用的截图类型",
                                                value=Setinitialize.screenshot_type)  # 单选
                    with gr.Row(): #并行显示，可开多列
                        with gr.Column(): # 并列显示，可开多行
                            msing_range = gr.Slider(1, 320, value=Setinitialize.msing_range, label="范围",
                                                     info="设置的范围值")  # 滑动条

                    with gr.Row():  # 并行显示，可开多列
                        with gr.Column(): # 并列显示，可开多行
                            with gr.Column():  # 并列显示，可开多行
                                with gr.Tab("移动设置"):
                                    with gr.Column():  # 并列显示，可开多行
                                        with gr.Row(elem_id='水平调整'):  # 并行显示，可开多列
                                            p_x = gr.Slider(0, 10, value=Setinitialize.p_x, label="水平拉抢速度",
                                                                info="值越高，拉抢速度越快，但是容易导致拉过头")  # 滑动条
                                            i_x = gr.Slider(0, 3, value=Setinitialize.i_x, label="水平动态补偿",
                                                                info="值越高，补偿速度越快，但是容易补偿过头")  # 滑动条
                                            d_x = gr.Slider(0, 10, value=Setinitialize.d_x, label="水平摩擦阻尼",
                                                                info="值越高，摩擦阻尼越大，适当调整该值")  # 滑动条
                                            c_x = gr.Slider(minimum=int(1), maximum=int(30), value=Setinitialize.c_x, label="水平平滑系数",
                                                                info="值越高，摩擦阻尼越大，适当调整该值")  # 滑动条
                                        with gr.Row():  # 并行显示，可开多列
                                            p_y = gr.Slider(0, 10, value=Setinitialize.p_y, label="垂直拉抢速度",
                                                                 info="值越高，拉抢速度越快，但是容易导致拉过头")  # 滑动条
                                            i_y = gr.Slider(0, 3, value=Setinitialize.i_y, label="垂直动态补偿",
                                                                 info="值越高，补偿速度越快，但是容易补偿过头")  # 滑动条
                                            d_y = gr.Slider(0, 10, value=Setinitialize.d_y, label="垂直摩擦阻尼",
                                                                 info="值越高，摩擦阻尼越大，适当调整该值")  # 滑动条
                                            c_y = gr.Slider(minimum=int(1), maximum=int(30), value=Setinitialize.c_y, label="垂直平滑系数",
                                                                 info="值越高，摩擦阻尼越大，适当调整该值")  # 滑动条
                            with gr.Row():  # 并列显示，可开多行
                                bottom4 = gr.Button(value="启动")
                                bottom15 = gr.Button(value="结束")
                                bottom9 = gr.Button(value="重启软件")

                                bottom9.click(restart_program)
                                bottom4.click(main)  # 触发
                                bottom15.click(jieshu)  # 触发
        with gr.Tab("其余设置"):
            with gr.Column():  # 并列显示，可开多行
                head_msing_ratio = gr.Slider(0, 1, value=Setinitialize.head_msing_ratio, label="头部—移动比例",
                                     info=None)  # 滑动条

                body_msing_ratio = gr.Slider(0, 1, value=Setinitialize.body_msing_ratio, label="身体—移动比例",
                                     info=None)  # 滑动条

                scr = gr.Slider(0, 1, value=Setinitialize.scr, label="置信度",
                                     info=None)  # 滑动条

                nms = gr.Slider(0, 1, value=Setinitialize.nms, label="交并比",
                                     info=None)  # 滑动条

        with gr.Tab("热键设置"):
            with gr.Tab("通用设置"):
                with gr.Column():  # 并列显示，可开多行
                    with gr.Row(): #并行显示，可开多列
                        eixt_key = gr.Dropdown(choices=Getinitialize.keys, value=Setinitialize.eixt_key, label="退出程序",
                                            info="快捷键退出程序")  # 单选
                        show_key = gr.Dropdown(choices=Getinitialize.keys, value=Setinitialize.show_key, label="显示截图",
                                            info="可以显示或关闭实时画面")  # 单选
                with gr.Tab("移动设置"):
                    with gr.Row():  # 并行显示，可开多列
                        target_key = gr.Dropdown(choices=Getinitialize.keys, value=Setinitialize.target_key, label="锁定目标",
                                            info="可以开关目标锁定功能")  # 单选
                        hb_key = gr.Dropdown(choices=Getinitialize.keys, value=Setinitialize.hb_key, label="头身切换",
                                            info="可以快捷切换头身，需设定好比例值")  # 单选
                        key_lock_key_1 = gr.Dropdown(choices=Getinitialize.keys, value=Setinitialize.key_lock_key_1,
                                                     label="长按移动-1", info="长按移动的快捷功能")  # 单选
                        key_lock_key_2 = gr.Dropdown(choices=Getinitialize.keys, value=Setinitialize.key_lock_key_2,
                                                     label="长按移动-2", info="长按移动的快捷功能")  # 单选

                    with gr.Row():  # 并行显示，可开多列
                        key_lock_key_3 = gr.Dropdown(choices=Getinitialize.keys, value=Setinitialize.key_lock_key_3,
                                                     label="开关移动", info="可以点击切换开关移动的热键")  # 单选
                        key_label_12 = gr.Dropdown(choices=Getinitialize.keys, value=Setinitialize.key_label_12,
                                                     label="启用双标签", info="可以开启两个选择标签")  # 单选
                        key_label_1 = gr.Dropdown(choices=Getinitialize.keys, label="启用①标签", value=Setinitialize.key_label_1, info='可以开启一号标签')
                        key_label_2 = gr.Dropdown(choices=Getinitialize.keys, label="启用②标签", value=Setinitialize.key_label_2, info='可以开启二号标签')

        inputs = [model, mobile_type, screenshot_type, head_msing_ratio, body_msing_ratio, scr, nms, msing_range,
                  p_x, i_x, d_x, c_x, p_y, i_y, d_y, c_y, eixt_key, show_key, target_key,
                  hb_key, key_lock_key_1, key_lock_key_2, key_lock_key_3, key_label_12,key_label_1,key_label_2]

        model.change(save, inputs=inputs)
        mobile_type.change(save, inputs=inputs)
        screenshot_type.change(save, inputs=inputs)
        msing_range.change(save, inputs=inputs)
        p_x.change(save, inputs=inputs)
        i_x.change(save, inputs=inputs)
        d_x.change(save, inputs=inputs)
        c_x.change(save, inputs=inputs)
        p_y.change(save, inputs=inputs)
        i_y.change(save, inputs=inputs)
        d_y.change(save, inputs=inputs)
        c_y.change(save, inputs=inputs)
        eixt_key.change(save, inputs=inputs)
        show_key.change(save, inputs=inputs)
        target_key.change(save, inputs=inputs)
        hb_key.change(save, inputs=inputs)
        key_lock_key_1.change(save, inputs=inputs)
        key_lock_key_2.change(save, inputs=inputs)
        key_lock_key_3.change(save, inputs=inputs)
        key_label_12.change(save, inputs=inputs)
        key_label_1.change(save, inputs=inputs)
        key_label_2.change(save, inputs=inputs)
        head_msing_ratio.change(save, inputs=inputs)
        body_msing_ratio.change(save, inputs=inputs)
        scr.change(save, inputs=inputs)
        nms.change(save, inputs=inputs)
    demo.launch(inbrowser=True)
