# Unitree G1 Real Robot
import time  # 导入时间模块
from contextlib import contextmanager  # 从上下文管理模块导入上下文管理器

import cv2  # 导入OpenCV库
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
import numpy as np  # 导入numpy库用于数值计算
import torch  # 导入PyTorch库
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # 从lerobot库导入LeRobotDataset类
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig  # 从lerobot库导入OpenCVCameraConfig类
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode  # 从lerobot库导入TorqueMode类
from lerobot.common.robot_devices.robots.configs import So100RobotConfig  # 从lerobot库导入So100RobotConfig类
from lerobot.common.robot_devices.robots.utils import make_robot_from_config  # 从lerobot库导入make_robot_from_config函数
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError  # 从lerobot库导入RobotDeviceAlreadyConnectedError异常

# NOTE:
# 有时我们希望抽象不同的环境，或者在单独的机器上运行
# 用户可以将这个单一的python类方法gr00t/eval/service.py
# 移动到他们的代码中，或者执行以下行
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from service import ExternalRobotInferenceClient  # 从service模块导入ExternalRobotInferenceClient类

# Import tqdm for progress bar
from tqdm import tqdm  # 导入tqdm库用于进度条

#################################################################################

class SO100Robot:
    def __init__(self, calibrate=False, enable_camera=False, camera_index=9):  # 初始化SO100Robot类
        self.config = So100RobotConfig()  # 创建So100RobotConfig配置
        self.calibrate = calibrate  # 是否校准
        self.enable_camera = enable_camera  # 是否启用相机
        self.camera_index = camera_index  # 相机索引
        if not enable_camera:  # 如果不启用相机
            self.config.cameras = {}  # 设置相机配置为空
        else:  # 如果启用相机
            self.config.cameras = {"webcam": OpenCVCameraConfig(camera_index, 30, 640, 480, "bgr")}  # 配置相机
        self.config.leader_arms = {}  # 设置领导臂配置

        # remove the .cache/calibration/so100 folder
        if self.calibrate:  # 如果需要校准
            import os  # 导入os模块
            import shutil  # 导入shutil模块

            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so100")  # 获取校准文件夹路径
            print("========> Deleting calibration_folder:", calibration_folder)  # 打印删除校准文件夹的信息
            if os.path.exists(calibration_folder):  # 如果校准文件夹存在
                shutil.rmtree(calibration_folder)  # 删除校准文件夹

        # Create the robot
        self.robot = make_robot_from_config(self.config)  # 根据配置创建机器人
        self.motor_bus = self.robot.follower_arms["main"]  # 获取主臂的电机总线

    @contextmanager
    def activate(self):  # 上下文管理器，用于激活机器人
        try:
            self.connect()  # 连接机器人
            self.move_to_initial_pose()  # 移动到初始姿势
            yield  # 允许执行上下文中的代码
        finally:
            self.disconnect()  # 断开连接

    def connect(self):  # 连接机器人
        if self.robot.is_connected:  # 如果机器人已经连接
            raise RobotDeviceAlreadyConnectedError(  # 抛出异常
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the arms
        self.motor_bus.connect()  # 连接电机总线

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)  # 禁用扭矩以进行校准

        # Calibrate the robot
        self.robot.activate_calibration()  # 激活机器人校准

        self.set_so100_robot_preset()  # 设置SO100机器人的预设

        # Enable torque on all motors of the follower arms
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)  # 启用电机的扭矩
        print("robot present position:", self.motor_bus.read("Present_Position"))  # 打印机器人的当前位置
        self.robot.is_connected = True  # 设置机器人为已连接状态

        self.camera = self.robot.cameras["webcam"] if self.enable_camera else None  # 获取相机
        if self.camera is not None:  # 如果相机存在
            self.camera.connect()  # 连接相机
        print("================> SO100 Robot is fully connected =================")  # 打印机器人已完全连接的信息

    def set_so100_robot_preset(self):  # 设置SO100机器人的预设
        # Mode=0 for Position Control
        self.motor_bus.write("Mode", 0)  # 设置模式为位置控制
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        # self.motor_bus.write("P_Coefficient", 16)
        self.motor_bus.write("P_Coefficient", 10)  # 设置P系数
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.motor_bus.write("I_Coefficient", 0)  # 设置I系数
        self.motor_bus.write("D_Coefficient", 32)  # 设置D系数
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.motor_bus.write("Lock", 0)  # 关闭写入锁
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.motor_bus.write("Maximum_Acceleration", 254)  # 设置最大加速度
        self.motor_bus.write("Acceleration", 254)  # 设置加速度

    def move_to_initial_pose(self):  # 移动到初始姿势
        current_state = self.robot.capture_observation()["observation.state"]  # 获取当前状态
        print("current_state", current_state)  # 打印当前状态
        # print all keys of the observation
        print("observation keys:", self.robot.capture_observation().keys())  # 打印观察的所有键

        current_state[0] = 90  # 设置关节0的目标角度
        current_state[2] = 90  # 设置关节2的目标角度
        current_state[3] = 90  # 设置关节3的目标角度
        self.robot.send_action(current_state)  # 发送当前状态
        time.sleep(2)  # 等待2秒

        current_state[4] = -70  # 设置关节4的目标角度
        current_state[5] = 30  # 设置关节5的目标角度
        current_state[1] = 90  # 设置关节1的目标角度
        self.robot.send_action(current_state)  # 发送当前状态
        time.sleep(2)  # 等待2秒

        print("----------------> SO100 Robot moved to initial pose")  # 打印机器人已移动到初始姿势的信息

    def go_home(self):  # 回到家位置
        # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        print("----------------> SO100 Robot moved to home pose")  # 打印机器人已移动到家位置的信息
        home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])  # 定义家位置的状态
        self.set_target_state(home_state)  # 设置目标状态为家位置
        time.sleep(2)  # 等待2秒

    def get_observation(self):  # 获取观察数据
        return self.robot.capture_observation()  # 返回观察数据

    def get_current_state(self):  # 获取当前状态
        return self.get_observation()["observation.state"].data.numpy()  # 返回当前状态的numpy数组

    def get_current_img(self):  # 获取当前图像
        img = self.get_observation()["observation.images.webcam"].data.numpy()  # 获取当前图像数据
        # convert bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式
        return img  # 返回图像

    def set_target_state(self, target_state: torch.Tensor):  # 设置目标状态
        self.robot.send_action(target_state)  # 发送目标状态

    def enable(self):  # 启用电机
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)  # 启用扭矩

    def disable(self):  # 禁用电机
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)  # 禁用扭矩

    def disconnect(self):  # 断开连接
        self.disable()  # 禁用电机
        self.robot.disconnect()  # 断开机器人连接
        self.robot.is_connected = False  # 设置机器人为未连接状态
        print("================> SO100 Robot disconnected")  # 打印机器人已断开连接的信息

    def __del__(self):  # 析构函数
        self.disconnect()  # 断开连接


#################################################################################

class Gr00tRobotInferenceClient:  # 定义Gr00tRobotInferenceClient类
    def __init__(self,  # 初始化Gr00tRobotInferenceClient类
        host="localhost",  # 主机地址
        port=5555,  # 端口号
        language_instruction="Pick up the fruits and place them on the plate.",  # 语言指令
    ):
        self.language_instruction = language_instruction  # 设置语言指令
        # 480, 640
        self.img_size = (480, 640)  # 设置图像大小
        self.policy = ExternalRobotInferenceClient(host=host, port=port)  # 创建外部机器人推理客户端

    def get_action(self, img, state):  # 获取动作
        obs_dict = {  # 创建观察字典
            "video.webcam": img[np.newaxis, :, :, :],  # 添加图像数据
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),  # 添加单臂状态
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),  # 添加夹爪状态
            "annotation.human.action.task_description": [self.language_instruction],  # 添加任务描述
        }
        start_time = time.time()  # 记录开始时间
        res = self.policy.get_action(obs_dict)  # 获取动作
        print("Inference query time taken", time.time() - start_time)  # 打印推理查询时间
        return res  # 返回动作

    def sample_action(self):  # 采样动作
        obs_dict = {  # 创建观察字典
            "video.webcam": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),  # 添加空图像数据
            "state.single_arm": np.zeros((1, 5)),  # 添加空单臂状态
            "state.gripper": np.zeros((1, 1)),  # 添加空夹爪状态
            "annotation.human.action.task_description": [self.language_instruction],  # 添加任务描述
        }
        return self.policy.get_action(obs_dict)  # 返回动作


#################################################################################

def view_img(img, img2=None):  # 定义视图图像函数
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    这是一个matplotlib查看器，因为cv2.imshow在lerobot env中可能是片状的
    还能够覆盖图像，以确保相机视图与训练设置对齐
    """
    plt.imshow(img)  # 显示图像
    if img2 is not None:  # 如果第二张图像存在
        plt.imshow(img2, alpha=0.5)  # 叠加显示第二张图像
    plt.axis("off")  # 关闭坐标轴
    plt.pause(0.001)  # 非阻塞显示
    plt.clf()  # 清除图形以便下一个帧


#################################################################################

if __name__ == "__main__":  # 如果该脚本是主程序
    import argparse  # 导入argparse模块
    import os  # 导入os模块

    default_dataset_path = os.path.expanduser("~/datasets/so100_strawberry_grape")  # 默认数据集路径

    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    parser.add_argument(
        "--use_policy", action="store_true"  # 默认是播放提供的数据集
    )
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)  # 数据集路径
    parser.add_argument("--host", type=str, default="10.110.17.183")  # 主机地址
    parser.add_argument("--port", type=int, default=5555)  # 端口号
    parser.add_argument("--action_horizon", type=int, default=12)  # 动作范围
    parser.add_argument("--actions_to_execute", type=int, default=350)  # 要执行的动作数量
    parser.add_argument("--camera_index", type=int, default=9)  # 相机索引
    args = parser.parse_args()  # 解析命令行参数

    ACTIONS_TO_EXECUTE = args.actions_to_execute  # 要执行的动作数量
    USE_POLICY = args.use_policy  # 是否使用策略
    ACTION_HORIZON = (
        args.action_horizon
    )  # 我们将只执行动作块中的一些动作
    MODALITY_KEYS = ["single_arm", "gripper"]  # 模态键

    if USE_POLICY:  # 如果使用策略
        client = Gr00tRobotInferenceClient(  # 创建Gr00tRobotInferenceClient实例
            host=args.host,  # 主机地址
            port=args.port,  # 端口号
            language_instruction="Pick up the fruits and place them on the plate.",  # 语言指令
        )

        robot = SO100Robot(calibrate=False, enable_camera=True, camera_index=args.camera_index)  # 创建SO100Robot实例
        with robot.activate():  # 激活机器人
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):  # 遍历要执行的动作
                img = robot.get_current_img()  # 获取当前图像
                view_img(img)  # 显示当前图像
                state = robot.get_current_state()  # 获取当前状态
                action = client.get_action(img, state)  # 获取动作
                start_time = time.time()  # 记录开始时间
                for i in range(ACTION_HORIZON):  # 遍历动作范围
                    concat_action = np.concatenate(  # 连接动作
                        [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],  # 获取每个模态的动作
                        axis=0,  # 沿着第0轴连接
                    )
                    assert concat_action.shape == (6,), concat_action.shape  # 确保连接后的动作形状正确
                    robot.set_target_state(torch.from_numpy(concat_action))  # 设置目标状态
                    time.sleep(0.01)  # 等待0.01秒

                    # get the realtime image
                    img = robot.get_current_img()  # 获取实时图像
                    view_img(img)  # 显示实时图像

                    # 0.05*16 = 0.8 seconds
                    print("executing action", i, "time taken", time.time() - start_time)  # 打印执行动作的时间
                print("Action chunk execution time taken", time.time() - start_time)  # 打印动作块执行时间
    else:  # 如果不使用策略
        # Test Dataset Source https://huggingface.co/datasets/youliangtan/so100_strawberry_grape
        dataset = LeRobotDataset(  # 创建LeRobotDataset实例
            repo_id="youliangtan/so100_strawberry_grape",  # 数据集ID
            root=args.dataset_path,  # 数据集路径
        )

        robot = SO100Robot(calibrate=False, enable_camera=True, camera_index=args.camera_index)  # 创建SO100Robot实例
        with robot.activate():  # 激活机器人
            actions = []  # 初始化动作列表
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Loading actions"):  # 遍历要加载的动作
                action = dataset[i]["action"]  # 获取动作
                img = dataset[i]["observation.images.webcam"].data.numpy()  # 获取图像数据
                # original shape (3, 480, 640) for image data
                realtime_img = robot.get_current_img()  # 获取实时图像

                img = img.transpose(1, 2, 0)  # 转置图像数据
                view_img(img, realtime_img)  # 显示图像和实时图像
                actions.append(action)  # 添加动作到列表

            # plot the actions
            plt.plot(actions)  # 绘制动作
            plt.show()  # 显示绘图

            print("Done initial pose")  # 打印初始姿势完成的信息

            # Use tqdm to create a progress bar
            for action in tqdm(actions, desc="Executing actions"):  # 遍历要执行的动作
                img = robot.get_current_img()  # 获取当前图像
                view_img(img)  # 显示当前图像

                robot.set_target_state(action)  # 设置目标状态为当前动作
                time.sleep(0.05)  # 等待0.05秒

            print("Done all actions")  # 打印所有动作完成的信息
            robot.go_home()  # 回到家位置
            print("Done home")  # 打印回到家位置完成的信息
