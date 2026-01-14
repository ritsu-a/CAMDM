import sys
sys.path.append('./')

import os
import pickle
import copy
import numpy as np
import utils.motion_modules as motion_modules
import style_helper as style100
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from utils.bvh_motion import Motion


class NPZMotion:
    """
    用于处理 NPZ 格式动作数据的类
    不需要 offsets、parents 等结构信息，因为 global_body_pos 已经包含了全局位置
    """
    def __init__(self, qpos, global_positions, names, root_quat=None, root_pos=None, filepath=None, frametime=1.0/60.0):
        """
        qpos: (frame_num, joint_angles_dim) - 关节角度数据（29个关节的角度）
        global_positions: (frame_num, joint_num, 3) - 全局位置
        names: 关节名称列表
        root_quat: (frame_num, 4) - 根关节的四元数 (wxyz格式)，可选
        root_pos: (frame_num, 3) - 根关节位置，可选
        """
        self.qpos = qpos
        self.global_positions = global_positions
        self.names = names
        self.root_quat = root_quat  # 根关节四元数
        self.root_pos = root_pos  # 根关节位置
        self.filepath = filepath
        self.frametime = frametime
        self.frame_num = qpos.shape[0]
        self.joint_num = len(names)
    
    def copy(self):
        return NPZMotion(
            qpos=copy.deepcopy(self.qpos),
            global_positions=copy.deepcopy(self.global_positions),
            names=self.names[:],
            root_quat=copy.deepcopy(self.root_quat) if self.root_quat is not None else None,
            root_pos=copy.deepcopy(self.root_pos) if self.root_pos is not None else None,
            filepath=self.filepath,
            frametime=self.frametime
        )
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return NPZMotion(
                qpos=self.qpos[index:index+1],
                global_positions=self.global_positions[index:index+1],
                names=self.names,
                root_quat=self.root_quat[index:index+1] if self.root_quat is not None else None,
                root_pos=self.root_pos[index:index+1] if self.root_pos is not None else None,
                filepath=self.filepath,
                frametime=self.frametime
            )
        elif isinstance(index, slice) or isinstance(index, np.ndarray):
            return NPZMotion(
                qpos=self.qpos[index],
                global_positions=self.global_positions[index],
                names=self.names,
                root_quat=self.root_quat[index] if self.root_quat is not None else None,
                root_pos=self.root_pos[index] if self.root_pos is not None else None,
                filepath=self.filepath,
                frametime=self.frametime
            )
        else:
            raise TypeError("Invalid argument type.")


def convert_z_up_to_y_up(positions, quaternions=None):
    """
    将坐标系从+Z朝上（OpenGL/Z-up）转换为+Y朝上（BVH/Y-up）
    
    转换规则：
    - X保持不变（右）
    - Y <- Z（上）
    - Z <- -Y（前）
    
    对于位置：(x, y, z) -> (x, z, -y)
    对于四元数：需要相应的旋转变换
    """
    # 位置转换：(x, y, z) -> (x, z, -y)
    converted_positions = np.zeros_like(positions)
    converted_positions[..., 0] = positions[..., 0]  # X保持不变
    converted_positions[..., 1] = positions[..., 2]  # Y <- Z（上）
    converted_positions[..., 2] = -positions[..., 1]  # Z <- -Y（前）
    
    if quaternions is not None:
        # 四元数转换：绕X轴旋转-90度，将Z-up转换为Y-up
        # 旋转轴是X轴，角度是-90度（顺时针）
        # 转换四元数（wxyz格式）
        convert_quat = R.from_euler('x', -np.pi/2).as_quat()  # xyzw格式
        
        converted_quaternions = np.zeros_like(quaternions)
        for i in range(quaternions.shape[0]):
            # 当前四元数（wxyz格式）转换为xyzw格式进行运算
            q_xyzw = quaternions[i][[1, 2, 3, 0]]
            # 应用坐标转换旋转
            q_rotated = (R.from_quat(convert_quat) * R.from_quat(q_xyzw)).as_quat()
            # 转回wxyz格式
            converted_quaternions[i] = q_rotated[[3, 0, 1, 2]]
        
        return converted_positions, converted_quaternions
    
    return converted_positions


def convert_y_up_to_z_up(positions, quaternions=None):
    """
    将坐标系从+Y朝上（BVH/Y-up）转换回+Z朝上（OpenGL/Z-up）
    这是 convert_z_up_to_y_up 的逆变换
    
    转换规则：
    - X保持不变（右）
    - Z <- Y（上）
    - Y <- -Z（前）
    
    对于位置：(x, y, z) -> (x, -z, y)
    对于四元数：需要相应的旋转变换（绕X轴旋转+90度）
    """
    # 位置转换：(x, y, z) -> (x, -z, y)
    converted_positions = np.zeros_like(positions)
    converted_positions[..., 0] = positions[..., 0]  # X保持不变
    converted_positions[..., 1] = -positions[..., 2]  # Y <- -Z（前）
    converted_positions[..., 2] = positions[..., 1]  # Z <- Y（上）
    
    if quaternions is not None:
        # 四元数转换：绕X轴旋转+90度，将Y-up转换回Z-up
        # 旋转轴是X轴，角度是+90度（逆时针），这是之前-90度的逆变换
        convert_quat = R.from_euler('x', np.pi/2).as_quat()  # xyzw格式
        
        converted_quaternions = np.zeros_like(quaternions)
        for i in range(quaternions.shape[0]):
            # 当前四元数（wxyz格式）转换为xyzw格式进行运算
            q_xyzw = quaternions[i][[1, 2, 3, 0]]
            # 应用坐标转换旋转
            q_rotated = (R.from_quat(convert_quat) * R.from_quat(q_xyzw)).as_quat()
            # 转回wxyz格式
            converted_quaternions[i] = q_rotated[[3, 0, 1, 2]]
        
        return converted_positions, converted_quaternions
    
    return converted_positions


def load_npz_motion(npz_path, start_idx=None, end_idx=None, apply_coordinate_transform=True):
    """
    从 NPZ 文件加载动作数据并创建 NPZMotion 对象
    NPZ 文件包含：
    - qpos: (frames, 36) - 关节位置和旋转
    - global_body_pos: (frames, 39, 3) - 全局身体位置
    - joint_names: (30,) - 关节名称
    
    注意：NPZ数据是+Z朝上，需要转换为+Y朝上以匹配BVH格式
    """
    data = np.load(npz_path)
    
    qpos = data['qpos']  # (frames, 36)
    global_body_pos = data['global_body_pos']  # (frames, 39, 3)
    joint_names = data['joint_names']  # (30,) - 第一个是 'root'，需要忽略
    jnt_type = data['jnt_type']  # (30,)
    
    frame_num = qpos.shape[0]
    
    # 忽略 joint_names 的第一个 'root'
    # 实际关节数量是 29 个
    actual_joint_names = joint_names[1:].tolist()  # 忽略第一个 'root'
    num_joints = len(actual_joint_names)  # 应该是 29
    
    # 切片处理
    if start_idx is not None or end_idx is not None:
        start_idx = start_idx if start_idx is not None else 0
        end_idx = end_idx if end_idx is not None else frame_num
        qpos = qpos[start_idx:end_idx]
        global_body_pos = global_body_pos[start_idx:end_idx]
        frame_num = qpos.shape[0]
    
    # qpos 格式：前3个是根位置，接下来4个是根旋转四元数，后面29个是关节角度
    # 提取根位置和根四元数
    root_pos = qpos[:, 0:3]  # (frames, 3) - 根位置
    root_quat = qpos[:, 3:7]  # (frames, 4) - 根旋转四元数，wxyz格式
    qpos_joint_angles = qpos[:, 7:]  # (frames, 29) - 29个关节的角度
    
    # global_body_pos 第一个位置对应 root，后面是29个关节的位置
    # 我们只取关节位置（忽略第一个 root 位置）
    if global_body_pos.shape[1] >= num_joints + 1:  # +1 是因为有 root
        # 忽略第一个位置（root），取后面的29个关节位置
        global_positions = global_body_pos[:, 1:num_joints+1, :]  # (frames, 29, 3)
    else:
        # 如果数量不够，尝试只取 num_joints 个
        global_positions = global_body_pos[:, 1:1+num_joints, :] if global_body_pos.shape[1] > 1 else global_body_pos
    
    # 如果需要进行坐标转换（从Z-up转换为Y-up）
    if apply_coordinate_transform:
        # 转换全局位置和根四元数
        global_positions, root_quat = convert_z_up_to_y_up(global_positions, root_quat)
        # 转换根位置
        root_pos = convert_z_up_to_y_up(root_pos.reshape(-1, 1, 3)).reshape(-1, 3)
    
    # frametime 设置为 1/60 (60fps)
    frametime = 1.0 / 60.0
    
    # 创建 NPZMotion 对象
    motion = NPZMotion(
        qpos=qpos_joint_angles,
        global_positions=global_positions,
        names=actual_joint_names,
        root_quat=root_quat,
        root_pos=root_pos,
        filepath=npz_path,
        frametime=frametime
    )
    
    return motion


def save_npzmotion_to_npz(motion, output_path, original_npz_data=None):
    """
    将 NPZMotion 对象保存为 NPZ 格式文件
    
    motion: NPZMotion 对象
    output_path: 输出文件路径
    original_npz_data: 原始npz数据（可选，用于保留其他字段如qvel, split_points等）
    """
    frame_num = motion.frame_num
    
    # 重建qpos: 前3个是根位置，接下来4个是根旋转四元数，后面29个是关节角度
    if motion.root_pos is not None:
        root_pos = motion.root_pos
    else:
        # 如果没有根位置，使用第一个关节的位置作为参考
        root_pos = motion.global_positions[:, 0, :]
    
    if motion.root_quat is not None:
        root_quat = motion.root_quat
    else:
        # 如果没有根四元数，使用单位四元数
        root_quat = np.zeros((frame_num, 4))
        root_quat[:, 0] = 1.0  # w=1
    
    # 重建完整的qpos (frames, 36)
    qpos = np.zeros((frame_num, 36))
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_quat
    qpos[:, 7:] = motion.qpos  # 29个关节的角度
    
    # 重建global_body_pos: 第一个是root，后面是29个关节
    global_body_pos = np.zeros((frame_num, 30, 3))  # root + 29个关节
    global_body_pos[:, 0, :] = root_pos  # root位置
    global_body_pos[:, 1:, :] = motion.global_positions  # 29个关节位置
    
    # 重建joint_names: 第一个是'root'，后面是29个关节名称
    joint_names = np.array(['root'] + motion.names)
    
    # 准备保存的数据
    save_dict = {
        'qpos': qpos,
        'global_body_pos': global_body_pos,
        'joint_names': joint_names,
    }
    
    # 如果提供了原始数据，保留其他字段
    if original_npz_data is not None:
        # 保留原始数据中的其他字段（如qvel, split_points, frequency, njnt, jnt_type）
        for key in ['qvel', 'split_points', 'frequency', 'njnt', 'jnt_type']:
            if key in original_npz_data:
                save_dict[key] = original_npz_data[key]
    
    # 如果没有提供原始数据，设置默认值
    if 'jnt_type' not in save_dict:
        # 根据joint_names创建jnt_type（默认为旋转关节）
        save_dict['jnt_type'] = np.zeros(len(joint_names), dtype=np.int32)
    
    if 'frequency' not in save_dict:
        save_dict['frequency'] = 1.0 / motion.frametime
    
    if 'njnt' not in save_dict:
        save_dict['njnt'] = len(joint_names)
    
    # 保存为npz文件
    np.savez(output_path, **save_dict)
    print(f'已保存NPZMotion到: {output_path}')


def load_contact(contact_path):
    contacts = []
    for line in open(contact_path, 'r'):
        contacts.append([int(val) for val in line.strip().split()])
    return np.array(contacts)


def mirror_text(statement):
    statement = statement.lower()
    temp_replacement = "_*temp*_"
    statement = statement.replace("left", temp_replacement)
    statement = statement.replace("right", "left")    
    statement = statement.replace(temp_replacement, "right")
    return statement


def extract_traj(root_positions, forwards, smooth_kernel=[5, 10]):
    traj_trans, traj_angles, traj_poses = [], [], []
    FORWARD_AXIS = np.array([[0, 0, 1]]) # Y-up system (Z-forward, Y-up, X-right)
    
    for kernel_size in smooth_kernel:
        smooth_traj = gaussian_filter1d(root_positions[:, [0, 2]], kernel_size, axis=0, mode='nearest')
        traj_trans.append(smooth_traj)
        
        forward = gaussian_filter1d(forwards, kernel_size, axis=0, mode='nearest')
        forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
        angle = np.arctan2(forward[:, 2], forward[:, 0])
        traj_angles.append(angle)
        
        v0s = FORWARD_AXIS.repeat(len(forward), axis=0)
        a = np.cross(v0s, forward)
        w = np.sqrt((v0s**2).sum(axis=-1) * (forward**2).sum(axis=-1)) + (v0s * forward).sum(axis=-1)
        between_wxyz = np.concatenate([w[...,np.newaxis], a], axis=-1)
        between = R.from_quat(between_wxyz[..., [1, 2, 3, 0]]).as_quat()[..., [3, 0, 1, 2]]
        
        traj_poses.append(between)
    
    return traj_trans, traj_angles, traj_poses
    
    
def extract_forward_hips_npz(motion, frame_idx, left_hip_name, right_hip_name, return_forward=False):
    """为 NPZMotion 提取前向方向"""
    if type(frame_idx) is int:
        frame_idx = [frame_idx]
    
    names = list(motion.names)
    try:
        l_h_idx, r_h_idx = names.index(left_hip_name), names.index(right_hip_name)
    except:
        raise Exception('Cannot find joint names, please check the names of Hips.')
    
    global_pos = motion.global_positions
    lower_across = global_pos[frame_idx, l_h_idx, :] - global_pos[frame_idx, r_h_idx, :]
    across = lower_across / np.sqrt((lower_across**2).sum(axis=-1))[...,np.newaxis]
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward_angle = np.arctan2(forward[:, 2], forward[:, 0])
    if return_forward:
        return forward_angle, forward
    return forward_angle


# 注意：NPZ数据已经以米为单位，不需要scaling操作
# def scaling_npz(motion, scaling_factor=0.01):
#     """缩放 NPZMotion（已废弃：NPZ数据已以米为单位）"""
#     motion = motion.copy()
#     motion.global_positions = motion.global_positions * scaling_factor
#     if motion.root_pos is not None:
#         motion.root_pos = motion.root_pos * scaling_factor
#     return motion


def root_npz(motion, given_pos=None):
    """将根位置重置到原点"""
    motion = motion.copy()
    root_init_pos = motion.root_pos[0, [0, 2]] if motion.root_pos is not None and given_pos is None else (motion.global_positions[0, 0, [0, 2]] if given_pos is None else given_pos)
    if motion.root_pos is not None:
        motion.root_pos[:, [0, 2]] -= root_init_pos
    motion.global_positions[:, :, [0, 2]] -= root_init_pos
    return motion


def on_ground_npz(motion):
    """
    将动作放到地面上
    注意：此函数假设数据已经是Y-up格式（Y轴是高度）
    """
    motion = motion.copy()
    global_pos = motion.global_positions
    lowest_height = np.min(global_pos[:, :, 1])  # Y轴是高度（Y-up格式）
    motion.global_positions[:, :, 1] -= lowest_height
    if motion.root_pos is not None:
        motion.root_pos[:, 1] -= lowest_height
    return motion


def mirror_npz(motion, l_joint_idxs, r_joint_idxs):
    """
    镜像 NPZMotion
    注意：此函数假设数据已经是Y-up格式
    
    motion: NPZMotion 对象
    l_joint_idxs: 左侧关节索引列表
    r_joint_idxs: 右侧关节索引列表
    
    重要：只对手臂关节（idx >= 15）的qpos取反，其他关节只交换左右
    """
    motion = motion.copy()
    ori_qpos = motion.qpos.copy()
    ori_global_positions = motion.global_positions.copy()
    
    # 镜像全局位置（X轴取反，Y-up格式）
    motion.global_positions[:, :, 0] *= -1
    
    # 交换左右关节的全局位置
    motion.global_positions[:, l_joint_idxs] = ori_global_positions[:, r_joint_idxs]
    motion.global_positions[:, r_joint_idxs] = ori_global_positions[:, l_joint_idxs]
    # 镜像后的位置需要再次镜像X轴
    motion.global_positions[:, l_joint_idxs, 0] *= -1
    motion.global_positions[:, r_joint_idxs, 0] *= -1
    
    # 镜像根位置（如果有）
    if motion.root_pos is not None:
        motion.root_pos[:, 0] *= -1
    
    # 镜像 qpos：交换左右关节的角度
    for l_idx, r_idx in zip(l_joint_idxs, r_joint_idxs):
        if l_idx < ori_qpos.shape[1] and r_idx < ori_qpos.shape[1]:
            # 交换左右关节的角度
            motion.qpos[:, l_idx] = ori_qpos[:, r_idx]
            motion.qpos[:, r_idx] = ori_qpos[:, l_idx]
    
    # 镜像角度：只对手臂关节（idx >= 15）取反角度值
    # 手臂关节索引从15开始（left_shoulder_pitch_joint等）
    arm_joint_start_idx = 15
    if motion.qpos.shape[1] > arm_joint_start_idx:
        # 对手臂关节（idx >= 15）取反
        motion.qpos[:, arm_joint_start_idx:] *= -1
    
    # 镜像根关节四元数（如果有）
    if motion.root_quat is not None:
        ori_root_quat = motion.root_quat.copy()
        # 镜像根旋转：Y-up格式下，y和z分量取反（wxyz格式，索引1,2,3是x,y,z）
        # 对于Y-up系统，镜像时需要调整四元数的y和z分量
        motion.root_quat[:, [2, 3]] *= -1  # y, z分量取反
    
    return motion


def process_npz_motion(motion):
    """
    处理 NPZMotion
    
    注意：NPZ数据已经以米为单位，不需要scaling操作
    """
    motion = root_npz(motion)
    motion = on_ground_npz(motion)
    
    # 尝试使用 g1 关节名称，如果不存在则使用 mixamo 名称
    try:
        _, forwards = extract_forward_hips_npz(motion, np.arange(motion.frame_num), style100.g1_left_hip_name, style100.g1_right_hip_name, return_forward=True)
    except:
        _, forwards = extract_forward_hips_npz(motion, np.arange(motion.frame_num), style100.left_hip_name, style100.right_hip_name, return_forward=True)
    
    root_positions = motion.root_pos if motion.root_pos is not None else motion.global_positions[:, 0]
    traj_trans, traj_angles, traj_poses = extract_traj(root_positions, forwards, smooth_kernel=[5, 10])
    
    # 直接保存 qpos（29个关节角度）和根关节四元数
    return motion, {
        'filepath': motion.filepath,
        'local_joint_rotations': motion.qpos,  # (frames, 29) - 29个关节的角度
        'root_quat': motion.root_quat,  # (frames, 4) - 根关节四元数 (wxyz格式)
        'global_root_positions': root_positions,
        'traj': traj_trans,
        'traj_angles': traj_angles,
        'traj_pose': traj_poses
    }


def process_motion(motion, use_scale=True):
    if use_scale:
        motion = motion_modules.scaling(motion, scaling_factor=0.01) # cm -> m
    motion = motion_modules.root(motion)
    motion = motion_modules.on_ground(motion)
    # _, forwards = motion_modules.extract_forward(motion, np.arange(motion.frame_num),
    #                                                         style100.left_shoulder_name, style100.right_shoulder_name, 
    #                                                         style100.left_hip_name, style100.right_hip_name, return_forward=True)
    _, forwards = motion_modules.extract_forward_hips(motion, np.arange(motion.frame_num), style100.left_hip_name, style100.right_hip_name, return_forward=True)
    traj_trans, traj_angles, traj_poses = extract_traj(motion.global_positions[:, 0], forwards, smooth_kernel=[5, 10])
    return motion, {
        'filepath': motion.filepath,
        'local_joint_rotations': motion.rotations,
        'global_root_positions': motion.global_positions[:, 0],
        'traj': traj_trans,
        'traj_angles': traj_angles,
        'traj_pose': traj_poses
    }


def test_npz_operations(npz_path, output_dir='data/test_output'):
    """
    测试NPZ的各种操作（mirror, on_ground, root等）
    所有操作都在Y-up格式的数据上进行，最终保存为Z-up格式
    
    npz_path: 输入的NPZ文件路径
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    basename = os.path.basename(npz_path).replace('.npz', '')
    original_data = np.load(npz_path)
    
    print(f'=== 测试NPZ操作: {basename} ===')
    
    # 1. 加载原始数据并转换为Y-up
    print('\n1. 加载并转换为Y-up格式...')
    motion_yup = load_npz_motion(npz_path, apply_coordinate_transform=True)
    print(f'   原始Y-up数据 - 第一帧根位置: {motion_yup.root_pos[0] if motion_yup.root_pos is not None else motion_yup.global_positions[0, 0]}')
    
    # 2. 测试 on_ground 操作
    print('\n2. 测试 on_ground 操作...')
    motion_yup_onground = on_ground_npz(motion_yup.copy())
    min_height_before = np.min(motion_yup.global_positions[:, :, 1])
    min_height_after = np.min(motion_yup_onground.global_positions[:, :, 1])
    print(f'   处理前最低高度: {min_height_before:.6f}')
    print(f'   处理后最低高度: {min_height_after:.6f} (应该接近0)')
    
    # 3. 测试 root 操作
    print('\n3. 测试 root 操作...')
    motion_yup_root = root_npz(motion_yup_onground.copy())
    root_pos_after = motion_yup_root.root_pos[0] if motion_yup_root.root_pos is not None else motion_yup_root.global_positions[0, 0]
    print(f'   根位置重置后: [{root_pos_after[0]:.6f}, {root_pos_after[1]:.6f}, {root_pos_after[2]:.6f}]')
    print(f'   X和Z应该接近0: X={root_pos_after[0]:.6f}, Z={root_pos_after[2]:.6f}')
    
    # 4. 测试 mirror 操作
    print('\n4. 测试 mirror 操作...')
    # 找到左右关节索引
    names = motion_yup_root.names
    l_names = sorted([val for val in names if 'left' in val.lower()])
    r_names = sorted([val for val in names if 'right' in val.lower()])
    
    if len(l_names) > 0 and len(r_names) > 0:
        l_joint_idxs = [names.index(name) for name in l_names]
        r_joint_idxs = [names.index(name) for name in r_names]
        
        # 保存镜像前的qpos用于验证
        original_qpos = motion_yup_root.qpos.copy()
        
        motion_yup_mirror = mirror_npz(motion_yup_root.copy(), l_joint_idxs, r_joint_idxs)
        
        # 验证镜像：左右关节位置应该交换
        original_l_pos = motion_yup_root.global_positions[0, l_joint_idxs[0]]
        original_r_pos = motion_yup_root.global_positions[0, r_joint_idxs[0]]
        mirror_l_pos = motion_yup_mirror.global_positions[0, l_joint_idxs[0]]
        mirror_r_pos = motion_yup_mirror.global_positions[0, r_joint_idxs[0]]
        
        print(f'   原始左侧关节位置: [{original_l_pos[0]:.3f}, {original_l_pos[1]:.3f}, {original_l_pos[2]:.3f}]')
        print(f'   原始右侧关节位置: [{original_r_pos[0]:.3f}, {original_r_pos[1]:.3f}, {original_r_pos[2]:.3f}]')
        print(f'   镜像后左侧关节位置: [{mirror_l_pos[0]:.3f}, {mirror_l_pos[1]:.3f}, {mirror_l_pos[2]:.3f}]')
        print(f'   镜像后右侧关节位置: [{mirror_r_pos[0]:.3f}, {mirror_r_pos[1]:.3f}, {mirror_r_pos[2]:.3f}]')
        print(f'   验证位置: 镜像后左侧应≈原始右侧（X取反）')
        print(f'            镜像后右侧应≈原始左侧（X取反）')
        
        # 验证qpos：只对手臂关节（idx >= 15）取反
        arm_joint_start_idx = 15
        num_joints = motion_yup_mirror.qpos.shape[1]
        
        print(f'\n   验证qpos（只对手臂关节idx >= {arm_joint_start_idx}取反）:')
        
        if num_joints > arm_joint_start_idx:
            # 找到左右手臂关节的索引（假设在l_joint_idxs和r_joint_idxs中）
            # 筛选出idx >= 15的关节（手臂关节）
            l_arm_idxs = [idx for idx in l_joint_idxs if idx >= arm_joint_start_idx]
            r_arm_idxs = [idx for idx in r_joint_idxs if idx >= arm_joint_start_idx]
            
            if len(l_arm_idxs) > 0 and len(r_arm_idxs) > 0:
                # 验证第一个左右手臂关节对
                l_arm_idx = l_arm_idxs[0]
                r_arm_idx = r_arm_idxs[0]
                
                original_l_arm_qpos = original_qpos[0, l_arm_idx]
                original_r_arm_qpos = original_qpos[0, r_arm_idx]
                mirror_l_arm_qpos = motion_yup_mirror.qpos[0, l_arm_idx]
                mirror_r_arm_qpos = motion_yup_mirror.qpos[0, r_arm_idx]
                
                print(f'   手臂关节验证 (left idx={l_arm_idx}, right idx={r_arm_idx}):')
                print(f'     原始左侧手臂qpos: {original_l_arm_qpos:.6f}')
                print(f'     原始右侧手臂qpos: {original_r_arm_qpos:.6f}')
                print(f'     镜像后左侧手臂qpos: {mirror_l_arm_qpos:.6f} (应为右侧取反: {-original_r_arm_qpos:.6f})')
                print(f'     镜像后右侧手臂qpos: {mirror_r_arm_qpos:.6f} (应为左侧取反: {-original_l_arm_qpos:.6f})')
                
                # 验证是否匹配（允许小的数值误差）
                if np.allclose(mirror_l_arm_qpos, -original_r_arm_qpos, atol=0.001):
                    print(f'     ✓ 左侧手臂关节验证通过: 镜像后 = -原始右侧')
                else:
                    print(f'     ✗ 左侧手臂关节验证失败')
                
                if np.allclose(mirror_r_arm_qpos, -original_l_arm_qpos, atol=0.001):
                    print(f'     ✓ 右侧手臂关节验证通过: 镜像后 = -原始左侧')
                else:
                    print(f'     ✗ 右侧手臂关节验证失败')
            
            # 验证非手臂关节（idx < 15）：只交换，不取反
            l_non_arm_idxs = [idx for idx in l_joint_idxs if idx < arm_joint_start_idx]
            r_non_arm_idxs = [idx for idx in r_joint_idxs if idx < arm_joint_start_idx]
            
            if len(l_non_arm_idxs) > 0 and len(r_non_arm_idxs) > 0:
                l_non_arm_idx = l_non_arm_idxs[0]
                r_non_arm_idx = r_non_arm_idxs[0]
                
                original_l_non_arm_qpos = original_qpos[0, l_non_arm_idx]
                original_r_non_arm_qpos = original_qpos[0, r_non_arm_idx]
                mirror_l_non_arm_qpos = motion_yup_mirror.qpos[0, l_non_arm_idx]
                mirror_r_non_arm_qpos = motion_yup_mirror.qpos[0, r_non_arm_idx]
                
                print(f'\n   非手臂关节验证 (left idx={l_non_arm_idx}, right idx={r_non_arm_idx}):')
                print(f'     原始左侧非手臂qpos: {original_l_non_arm_qpos:.6f}')
                print(f'     原始右侧非手臂qpos: {original_r_non_arm_qpos:.6f}')
                print(f'     镜像后左侧非手臂qpos: {mirror_l_non_arm_qpos:.6f} (应为右侧: {original_r_non_arm_qpos:.6f})')
                print(f'     镜像后右侧非手臂qpos: {mirror_r_non_arm_qpos:.6f} (应为左侧: {original_l_non_arm_qpos:.6f})')
                
                # 验证是否匹配（非手臂关节只交换，不取反）
                if np.allclose(mirror_l_non_arm_qpos, original_r_non_arm_qpos, atol=0.001):
                    print(f'     ✓ 左侧非手臂关节验证通过: 镜像后 = 原始右侧（仅交换）')
                else:
                    print(f'     ✗ 左侧非手臂关节验证失败')
                
                if np.allclose(mirror_r_non_arm_qpos, original_l_non_arm_qpos, atol=0.001):
                    print(f'     ✓ 右侧非手臂关节验证通过: 镜像后 = 原始左侧（仅交换）')
                else:
                    print(f'     ✗ 右侧非手臂关节验证失败')
        else:
            print(f'   警告: 关节数量 ({num_joints}) <= {arm_joint_start_idx}，无法验证手臂关节')
    else:
        print('   警告: 未找到左右关节，跳过mirror测试')
        motion_yup_mirror = motion_yup_root.copy()
    
    # 5. 将处理后的数据转换回Z-up并保存
    print('\n5. 转换回Z-up格式并保存...')
    motion_zup_processed = motion_yup_mirror.copy()
    
    # 转换全局位置
    motion_zup_processed.global_positions = convert_y_up_to_z_up(motion_yup_mirror.global_positions)
    
    # 转换根位置和根四元数
    if motion_yup_mirror.root_pos is not None:
        motion_zup_processed.root_pos, motion_zup_processed.root_quat = convert_y_up_to_z_up(
            motion_yup_mirror.root_pos.reshape(-1, 1, 3),
            motion_yup_mirror.root_quat
        )
        motion_zup_processed.root_pos = motion_zup_processed.root_pos.reshape(-1, 3)
    elif motion_yup_mirror.root_quat is not None:
        root_pos_yup = motion_yup_mirror.global_positions[:, 0:1, :]
        _, motion_zup_processed.root_quat = convert_y_up_to_z_up(root_pos_yup, motion_yup_mirror.root_quat)
        motion_zup_processed.root_pos = motion_zup_processed.global_positions[:, 0, :]
    
    # 保存处理后的数据
    processed_path = os.path.join(output_dir, f'{basename}_processed.npz')
    save_npzmotion_to_npz(motion_zup_processed, processed_path, original_data)
    print(f'   已保存处理后的数据（Z-up）: {processed_path}')
    
    # 6. 保存原始数据作为对比
    motion_raw = load_npz_motion(npz_path, apply_coordinate_transform=False)
    raw_path = os.path.join(output_dir, f'{basename}_raw.npz')
    save_npzmotion_to_npz(motion_raw, raw_path, original_data)
    print(f'   已保存原始数据（Z-up）: {raw_path}')
    
    print('\n=== 测试完成 ===')
    print(f'所有文件保存在: {output_dir}')
    print(f'  - {basename}_raw.npz (原始Z-up格式)')
    print(f'  - {basename}_processed.npz (处理后Z-up格式: on_ground + root + mirror)')
    
    return motion_raw, motion_zup_processed


def test_npz_coordinate_transform(npz_path, output_dir='data/test_output'):
    """
    测试NPZ坐标转换功能，保存处理前后的NPZMotion
    
    npz_path: 输入的NPZ文件路径
    output_dir: 输出目录（raw和after会保存在同一文件夹）
    
    注意：最终保存的所有npz文件都是Z-up格式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始NPZ数据（不进行坐标转换，Z-up格式）
    print(f'加载原始NPZ数据（Z-up格式）: {npz_path}')
    motion_before = load_npz_motion(npz_path, apply_coordinate_transform=False)
    
    # 加载并转换为Y-up格式进行处理
    print(f'加载并转换NPZ数据为Y-up格式进行处理: {npz_path}')
    motion_after_yup = load_npz_motion(npz_path, apply_coordinate_transform=True)
    
    # 将处理后的Y-up数据转换回Z-up格式以便保存
    print(f'将处理后的数据转换回Z-up格式...')
    motion_after_zup = motion_after_yup.copy()
    
    # 转换全局位置
    motion_after_zup.global_positions = convert_y_up_to_z_up(motion_after_yup.global_positions)
    
    # 转换根位置和根四元数
    if motion_after_yup.root_pos is not None:
        motion_after_zup.root_pos, motion_after_zup.root_quat = convert_y_up_to_z_up(
            motion_after_yup.root_pos.reshape(-1, 1, 3),
            motion_after_yup.root_quat
        )
        motion_after_zup.root_pos = motion_after_zup.root_pos.reshape(-1, 3)
    elif motion_after_yup.root_quat is not None:
        # 如果没有root_pos但有root_quat，从第一个关节位置提取并转换
        root_pos_yup = motion_after_yup.global_positions[:, 0:1, :]
        _, motion_after_zup.root_quat = convert_y_up_to_z_up(root_pos_yup, motion_after_yup.root_quat)
        # root_pos从转换后的global_positions提取
        motion_after_zup.root_pos = motion_after_zup.global_positions[:, 0, :]
    
    # 保存文件（放在同一文件夹）
    basename = os.path.basename(npz_path).replace('.npz', '')
    before_path = os.path.join(output_dir, f'{basename}_raw.npz')
    after_path = os.path.join(output_dir, f'{basename}_after.npz')
    
    # 加载原始npz数据以便保留其他字段
    original_data = np.load(npz_path)
    
    print(f'保存原始NPZMotion（Z-up）: {before_path}')
    save_npzmotion_to_npz(motion_before, before_path, original_data)
    
    print(f'保存处理后的NPZMotion（已转换回Z-up）: {after_path}')
    save_npzmotion_to_npz(motion_after_zup, after_path, original_data)
    
    # 打印一些对比信息
    print('\n=== 坐标转换对比 ===')
    print(f'原始数据（Z-up）第一帧第一个关节位置: {motion_before.global_positions[0, 0]}')
    print(f'处理后数据（Z-up）第一帧第一个关节位置: {motion_after_zup.global_positions[0, 0]}')
    print(f'中间处理（Y-up）第一帧第一个关节位置: {motion_after_yup.global_positions[0, 0]}')
    
    if motion_before.root_pos is not None and motion_after_zup.root_pos is not None:
        print(f'原始数据（Z-up）根位置: {motion_before.root_pos[0]}')
        print(f'处理后数据（Z-up）根位置: {motion_after_zup.root_pos[0]}')
    
    print(f'\n所有文件已保存到: {output_dir}')
    print(f'  - {basename}_raw.npz (原始Z-up格式)')
    print(f'  - {basename}_after.npz (处理后Z-up格式)')
    
    return motion_before, motion_after_zup

if __name__ == '__main__':
    
    # 测试NPZ坐标转换功能
    # 使用方法：设置 TEST_NPZ_FILE 环境变量或取消下面的注释
    import sys
    test_npz_path = os.getenv('TEST_NPZ_FILE', None)
    
    # 如果提供了测试文件路径，运行测试
    if test_npz_path and os.path.exists(test_npz_path):
        print('=== 开始测试NPZ坐标转换 ===')
        test_npz_coordinate_transform(test_npz_path, output_dir='data/test_output')
        print('=== 测试完成 ===\n')
        sys.exit(0)
    
    # 如果没有指定测试文件，尝试使用默认路径（可选）
    # 取消下面的注释以启用自动测试
    # default_test_path = 'data/100STYLE_g1/output_raw/Akimbo_FR_120Hz_29dof.npz'
    # if os.path.exists(default_test_path):
    #     print('=== 开始测试NPZ坐标转换（使用默认路径） ===')
    #     test_npz_coordinate_transform(default_test_path, output_dir='data/test_output')
    #     print('=== 测试完成 ===\n')
    
    data_root = 'data/100STYLE_g1'
    process_batch = os.path.join(data_root, 'output_raw')
    export_path = 'data/pkls/100style_g1.pkl'
    style_metas = style100.get_info(data_root, meta_file='Dataset_List.csv')
    
    data_list = {
        'parents': None,
        'offsets': None,
        'names': None,
        'motions': []
    }
    
    npz_files = []
    for root, dirs, files in os.walk(process_batch):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    npz_files = sorted(npz_files)
    
    for npz_path in tqdm(npz_files):
        
        # 文件名格式：StyleName_ActionLabel_120Hz_29dof.npz
        basename = os.path.basename(npz_path).replace('.npz', '')
        parts = basename.split('_')
        
        # 找到动作标签（BR, BW, FR, FW, ID, SR, SW, TR1, TR2, TR3）
        action_labels = ['BR', 'BW', 'FR', 'FW', 'ID', 'SR', 'SW', 'TR1', 'TR2', 'TR3']
        action_label = None
        style_name = None
        
        for label in action_labels:
            if label in parts:
                action_label = label
                # 动作标签之前的部分是 style_name
                label_idx = parts.index(label)
                style_name = '_'.join(parts[:label_idx])
                break
        
        if action_label is None or style_name is None:
            print(f'Warning: Cannot parse filename {npz_path}, skipping...')
            continue
        
        if style_name not in style_metas.keys():
            continue
        
        meta_info = style_metas[style_name]
        start_idx, end_idx = meta_info['framecuts'][action_label]
        
        # 检查framecuts是否有效
        if start_idx is None or end_idx is None:
            print(f'Warning: Invalid framecuts for {style_name}_{action_label}, skipping...')
            continue
        
        # 对于npz数据，framecuts可能需要调整（如果原始数据是120Hz，可能需要除以2）
        # 但根据npz文件名显示是120Hz，这里先不做调整，根据实际情况可能需要修改
        # start_idx, end_idx = start_idx // 2, end_idx // 2
        
        # 加载npz数据（转换为Y-up格式以便处理）
        motion = load_npz_motion(npz_path, start_idx=start_idx, end_idx=end_idx, apply_coordinate_transform=True)
        motion, motion_data = process_npz_motion(motion)
        motion_data['text'] = meta_info['description'].lower()
        motion_data['style'] = style_name
        data_list['motions'].append(motion_data)
        
        # NPZ数据不需要处理镜像
        # if meta_info['is_symmetric']:
        #     mirror_motion = motion.copy()
        #     names = mirror_motion.names
        #     l_names = sorted([val for val in names if 'left' in val.lower()])
        #     r_names = sorted([val for val in names if 'right' in val.lower()])
        #     l_joint_idxs, r_joint_idxs = [names.index(name) for name in l_names], [names.index(name) for name in r_names]
        #     mirror_motion = mirror_npz(mirror_motion, l_joint_idxs, r_joint_idxs)
        #     _, mirror_motion_data = process_npz_motion(mirror_motion)
        #     mirror_motion_data['filepath'] = mirror_motion_data['filepath'].replace('.npz', '.mirror.npz')
        #     mirror_motion_data['text'] = mirror_text(meta_info['description'])
        #     mirror_motion_data['style'] = style_name
        #     data_list['motions'].append(mirror_motion_data)
        print('Finish processing %s' % npz_path)
    
    # NPZ数据不需要T-pose
    # data_list['T_pose'] = None
    
    pickle.dump(data_list, open(export_path, 'wb'))
    print('Finish exporting %s' % export_path)
