#!/usr/bin/env python

"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py -p lerobot/diffusion_pusht eval.n_episodes=10
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.

```
python lerobot/scripts/eval.py \
    -p outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    eval.n_episodes=10
```

Note that in both examples, the repo/folder should contain at least `config.json`, `config.yaml` and
`model.safetensors`.

Note the formatting for providing the number of episodes. Generally, you may provide any number of arguments
with `qualified.parameter.name=value`. In this case, the parameter eval.n_episodes appears as `n_episodes`
nested under `eval` in the `config.yaml` found at
https://huggingface.co/lerobot/diffusion_pusht/tree/main.

请注意用于提供剧集数的格式。通常，您可以提供任意数量的参数
具有`qualified.parameter.name=value`。在这种情况下，参数eval.n_septs显示为`n_septs`
嵌套在位于的“config.yaml”中的“eval”下
https://huggingface.co/lerobot/diffusion_pusht/tree/main.
"""

# 导入所需的Python库和模块
import argparse  # 用于解析命令行参数
import json  # 用于JSON数据处理
import logging  # 用于日志记录
import threading  # 用于线程管理
import time  # 用于时间相关操作
from contextlib import nullcontext  # 用于上下文管理
from copy import deepcopy  # 用于深度复制对象
from datetime import datetime as dt  # 用于日期时间处理
from pathlib import Path  # 用于路径操作
from typing import Callable  # 用于类型注解

# 导入数据处理和机器学习相关的库
import einops  # 用于张量操作
import gymnasium as gym  # 用于强化学习环境
import numpy as np  # 用于数值计算
import torch  # 用于深度学习
from datasets import Dataset, Features, Image, Sequence, Value, concatenate_datasets  # 用于数据集处理
from huggingface_hub import snapshot_download  # 用于从HuggingFace下载模型
from huggingface_hub.utils._errors import RepositoryNotFoundError  # 用于处理仓库未找到错误

# 导入图像处理和模型相关的库
from huggingface_hub.utils._validators import HFValidationError  # 用于HuggingFace验证错误处理
from PIL import Image as PILImage  # 用于图像处理
from torch import Tensor, nn  # 用于神经网络操作
from tqdm import trange  # 用于进度条显示

# 导入自定义模块
from lerobot.common.datasets.factory import make_dataset  # 用于创建数据集
from lerobot.common.datasets.utils import hf_transform_to_torch  # 用于数据转换
from lerobot.common.envs.factory import make_env  # 用于创建环境
from lerobot.common.envs.utils import preprocess_observation  # 用于预处理观察数据
from lerobot.common.logger import log_output_dir  # 用于日志输出目录
from lerobot.common.policies.factory import make_policy  # 用于创建策略
from lerobot.common.policies.policy_protocol import Policy  # 用于策略协议
from lerobot.common.policies.utils import get_device_from_parameters  # 用于获取设备参数
from lerobot.common.utils.io_utils import write_video  # 用于视频写入
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed  # 用于各种工具函数
import cv2  # 用于计算机视觉处理

# 导入系统和路径相关的模块
from pathlib import Path  # 用于路径操作
import sys  # 用于系统操作

# 设置项目根目录和系统路径
project_root = Path(__file__).resolve().parents[3]  # 获取项目根目录
sys.path.append(str(project_root))  # 将项目根目录添加到系统路径

# 导入机器人控制相关的模块
from unitree_utils.image_server.image_client import ImageClient  # 用于图像客户端
from unitree_utils.robot_control.robot_arm import G1_29_ArmController  # 用于机器人手臂控制
from unitree_utils.robot_control.robot_hand_unitree import Dex3_1_Controller  # 用于机器人手部控制
from multiprocessing import Process, shared_memory, Array  # 用于多进程和共享内存

def get_image_processed(cam, img_size=[640, 480]):
    """处理相机图像的函数"""
    curr_images = []  # 创建空列表存储图像
    color_img  = cam  # 获取相机图像
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式
    color_img = cv2.resize(color_img, img_size)  # 调整图像大小
    curr_images.append(color_img)  # 将处理后的图像添加到列表
    color_img = np.stack(curr_images, axis=0)  # 将图像堆叠
    return color_img  # 返回处理后的图像

def eval_policy(
    policy: torch.nn.Module,  # 输入策略模型
) -> dict:  # 返回字典类型的评估结果
    """评估策略的主函数"""
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."  # 确保策略是PyTorch模块
    device = get_device_from_parameters(policy)  # 获取设备参数

    # 重置策略
    policy.reset()
    is_single_hand = True  # 是否使用单手模式
    use_left_hand = True  # 是否使用左手

    # 初始化机器人控制器
    g1_arm = G1_29_ArmController()  # 初始化机器人手臂控制器
    tirhand = Dex3_1_Controller()  # 初始化机器人手部控制器

    # 设置图像参数
    image_h = 480  # 图像高度
    image_w = 640  # 图像宽度
    img_shape = (image_h, image_w*2, 3)  # 图像形状
    img_shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)  # 创建共享内存
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=img_shm.buf)  # 创建图像数组
    img_client = ImageClient(img_shape = img_shape, img_shm_name = img_shm.name)  # 创建图像客户端
    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)  # 创建图像接收线程
    image_receive_thread.start()  # 启动图像接收线程

    # 初始化机器人
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")  # 等待用户输入启动信号
    if user_input.lower() == 's':
        q_pose=np.zeros(14)  # 初始化位置数组
        q_tau_ff=np.zeros(14)  # 初始化力矩数组

        print(f"init robot pose")  # 打印初始化信息
        # 设置目标位置
        targetPos = np.array([...], dtype=np.float32)  # 设置目标位置数组
        q_pose = targetPos  # 更新位置
        g1_arm.ctrl_dual_arm(q_pose, q_tau_ff)  # 控制双臂

        # 设置手部动作
        left_hand_action = np.array([...], dtype=np.float32)  # 设置左手动作
        right_hand_action = np.array([...], dtype=np.float32)  # 设置右手动作
        tirhand.ctrl(left_hand_action,right_hand_action)  # 控制手部
        print(f"wait robot to pose")  # 打印等待信息
        time.sleep(5)  # 等待5秒

        # 设置手臂动作
        left_arm_action = np.array([...], dtype=np.float32)  # 设置左臂动作
        right_arm_action = np.array([...], dtype=np.float32)  # 设置右臂动作

        # 设置控制频率
        frequency = 15.0  # 设置频率为15Hz
        period = 1.0 / frequency  # 计算周期
        i=0  # 初始化计数器
        next_time = time.time()  # 获取当前时间
        while i in range(100000):  # 主循环
            observation={}  # 初始化观察字典
            img_dic=dict()  # 初始化图像字典

            # 获取机器人状态
            armstate = g1_arm.get_current_dual_arm_q()  # 获取手臂状态
            handstate = tirhand.get_current_dual_hand_q()  # 获取手部状态

            # 处理状态数据
            if is_single_hand:  # 单手模式处理
                if use_left_hand:  # 使用左手
                    leftarmstate = armstate[:7]   
                    lefthandstate =handstate[:7]
                    qpos_data_processed = np.concatenate([leftarmstate,lefthandstate])
                else:  # 使用右手
                    rightarmstate = armstate[-7:]   
                    righthandstate =handstate[-7:]
                    qpos_data_processed = np.concatenate([rightarmstate,righthandstate])
            else:  # 双手模式处理
                qpos_data_processed = np.concatenate([armstate,handstate])

            # 打印位置信息
            print(f"qpose:{np.round(qpos_data_processed / np.pi * 180, 1)}")

            # 获取并处理图像
            current_image = img_array.copy()  # 复制当前图像
            left_image =  current_image[:, :image_w]  # 获取左图像
            right_image = current_image[:, image_w:]  # 获取右图像
            img_dic['top'] = get_image_processed(left_image)  # 处理左图像
            img_dic['wrist'] = get_image_processed(right_image)  # 处理右图像
            robot_state = qpos_data_processed  # 获取机器人状态
            observation["pixels"]= img_dic  # 存储图像数据
            observation["agent_pos"] = robot_state  # 存储位置数据

            # 预处理观察数据
            observation = preprocess_observation(observation)  # 预处理观察数据
            observation['observation.state'] =observation['observation.state'].unsqueeze(0)  # 增加维度
            observation = {key: observation[key].to(device, non_blocking=True) for key in observation}  # 将数据移到设备上

            # 使用策略选择动作
            with torch.inference_mode():
                action = policy.select_action(observation)  # 选择动作

            # 处理动作数据
            action = action.squeeze(0).to("cpu").numpy()  # 将动作转换为numpy数组
            print(f"qpose:{np.round(action / np.pi * 180, 1)}")  # 打印动作信息

            # 执行动作
            if is_single_hand:  # 单手模式执行
                if use_left_hand:  # 使用左手
                    left_arm_action = action[:7]
                    left_hand_action = action[-7:]
                    q_pose = np.concatenate([left_arm_action,right_arm_action],axis=0)
                    q_pose[3] = q_pose[3] - 0.12
                else:  # 使用右手
                    right_arm_action = action[:7]
                    right_hand_action = action[-7:]
                    q_pose = np.concatenate([left_arm_action,right_arm_action],axis=0)
                    q_pose[3+7] = q_pose[3+7] - 0.12
                
            else:  # 双手模式执行
                arm_action = action[:14]
                left_hand_action = action[14:14+7]
                right_hand_action = action[-(14+7):]
                q_pose = arm_action
                q_pose[3] = q_pose[3] - 0.12
                q_pose[3+7] = q_pose[3+7] - 0.12

            # 控制机器人执行动作
            g1_arm.ctrl_dual_arm(q_pose, q_tau_ff)  # 控制双臂
            tirhand.ctrl(left_hand_action,right_hand_action)  # 控制手部

            # 控制执行频率
            next_time += period  # 计算下一次执行时间
            sleep_time = next_time - time.time()  # 计算需要睡眠的时间
            if sleep_time > 0:  # 如果需要等待
                time.sleep(sleep_time)  # 等待
            else:  # 如果执行时间过长
                print("Warning: execution time exceeded the desired period")  # 打印警告


def main(
    pretrained_policy_path: Path | None = None,  # 预训练策略路径
    hydra_cfg_path: str | None = None,  # hydra配置路径
    out_dir: str | None = None,  # 输出目录
    config_overrides: list[str] | None = None,  # 配置覆盖
):
    """主函数，用于初始化和运行评估"""
    
    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)  # 确保只提供一个路径
    if pretrained_policy_path is not None:  # 如果提供了预训练策略路径
        hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), config_overrides)  # 初始化配置
    else:  # 如果提供了hydra配置路径
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)  # 初始化配置

    if out_dir is None:  # 如果没有指定输出目录
        out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"  # 创建输出目录

    # 检查并设置设备
    device = get_safe_torch_device(hydra_cfg.device, log=True)  # 获取安全的torch设备

    # 设置PyTorch后端
    torch.backends.cudnn.benchmark = True  # 启用cudnn基准测试
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许使用TF32
    set_global_seed(hydra_cfg.seed)  # 设置全局随机种子

    log_output_dir(out_dir)  # 记录输出目录

    logging.info("Making environment.")  # 记录创建环境信息
    env = make_env(hydra_cfg)  # 创建环境

    logging.info("Making policy.")  # 记录创建策略信息
    if hydra_cfg_path is None:  # 如果没有提供hydra配置路径
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))  # 创建策略
    else:  # 如果提供了hydra配置路径
        policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)  # 创建策略

    assert isinstance(policy, nn.Module)  # 确保策略是nn.Module类型
    policy.eval()  # 设置策略为评估模式

    # 执行评估
    with torch.no_grad(), torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext():
        info = eval_policy(
            policy
        )
    print(info["aggregated"])  # 打印聚合信息

    # 保存评估信息
    with open(Path(out_dir) / "eval_info.json", "w") as f:  # 打开文件
        json.dump(info, f, indent=2)  # 保存信息

    env.close()  # 关闭环境

    logging.info("End of eval")  # 记录评估结束信息


if __name__ == "__main__":  # 如果作为主程序运行
    init_logging()  # 初始化日志

    parser = argparse.ArgumentParser(  # 创建参数解析器
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=False)  # 创建互斥参数组
    group.add_argument(  # 添加预训练策略路径参数
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
       default='/home/unitree/Videos/lw/21-53-27_real_world_act_default/checkpoints/100000/pretrained_model'
    )
    group.add_argument(  # 添加配置文件路径参数
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")  # 添加版本ID参数
    parser.add_argument(  # 添加输出目录参数
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}_{env_name}_{policy_name}"
        ),
    )
    parser.add_argument(  # 添加配置覆盖参数
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()  # 解析命令行参数

    if args.pretrained_policy_name_or_path is None:  # 如果没有提供预训练策略路径
        main(hydra_cfg_path=args.config, out_dir=args.out_dir, config_overrides=args.overrides)  # 使用配置文件运行
    else:  # 如果提供了预训练策略路径
        try:  # 尝试下载模型
            pretrained_policy_path = Path(
                snapshot_download(args.pretrained_policy_name_or_path, revision=args.revision)
            )
        except (HFValidationError, RepositoryNotFoundError) as e:  # 处理错误
            if isinstance(e, HFValidationError):
                error_message = (
                    "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
                )
            else:
                error_message = (
                    "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
                )

            logging.warning(f"{error_message} Treating it as a local directory.")  # 记录警告信息
            pretrained_policy_path = Path(args.pretrained_policy_name_or_path)  # 使用本地路径
        if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():  # 检查路径是否有效
            raise ValueError(  # 抛出错误
                "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
                "repo ID, nor is it an existing local directory."
            )

        main(  # 运行主函数
            pretrained_policy_path=pretrained_policy_path,
            out_dir=None,
            config_overrides=None,
        )
