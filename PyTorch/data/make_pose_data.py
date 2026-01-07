import sys
sys.path.append('./')

import os
import pickle
import numpy as np
import utils.motion_modules as motion_modules
import style_helper as style100
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from utils.bvh_motion import Motion
import copy


class NPZMotion:
    """
    用于处理 NPZ 格式动作数据的类
    不需要 offsets、parents 等结构信息，因为 global_body_pos 已经包含了全局位置
    """
    def __init__(self, qpos, global_positions, names, root_quat=None, filepath=None, frametime=1.0/60.0):
        """
        qpos: (frame_num, joint_angles_dim) - 关节角度数据（29个关节的角度）
        global_positions: (frame_num, joint_num, 3) - 全局位置
        names: 关节名称列表
        root_quat: (frame_num, 4) - 根关节的四元数 (wxyz格式)，可选
        """
        self.qpos = qpos
        self.global_positions = global_positions
        self.names = names
        self.root_quat = root_quat  # 根关节四元数
        self.filepath = filepath
        self.frametime = frametime
        self.frame_num = qpos.shape[0]
        self.joint_num = len(names)
        
        # 计算 rotations（从 qpos 转换而来，用于兼容性）
        self._rotations = None
    
    def copy(self):
        return NPZMotion(
            qpos=copy.deepcopy(self.qpos),
            global_positions=copy.deepcopy(self.global_positions),
            names=self.names[:],
            root_quat=copy.deepcopy(self.root_quat) if self.root_quat is not None else None,
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
                filepath=self.filepath,
                frametime=self.frametime
            )
        elif isinstance(index, slice) or isinstance(index, np.ndarray):
            return NPZMotion(
                qpos=self.qpos[index],
                global_positions=self.global_positions[index],
                names=self.names,
                root_quat=self.root_quat[index] if self.root_quat is not None else None,
                filepath=self.filepath,
                frametime=self.frametime
            )
        else:
            raise TypeError("Invalid argument type.")


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


def load_npz_motion(npz_path, start_idx=None, end_idx=None):
    """
    从 NPZ 文件加载动作数据并创建 NPZMotion 对象
    NPZ 文件包含：
    - qpos: (frames, 36) - 关节位置和旋转
    - global_body_pos: (frames, 39, 3) - 全局身体位置
    - joint_names: (30,) - 关节名称
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
    # 我们只取关节角度部分（qpos[:, 7:]）对应29个关节
    qpos_joint_angles = qpos[:, 7:]  # (frames, 29) - 29个关节的角度
    
    # global_body_pos 第一个位置对应 root，后面是29个关节的位置
    # 我们只取关节位置（忽略第一个 root 位置）
    # 假设 global_body_pos 的形状是 (frames, 39, 3)，第一个是 root，后面是29个关节
    if global_body_pos.shape[1] >= num_joints + 1:  # +1 是因为有 root
        # 忽略第一个位置（root），取后面的29个关节位置
        global_positions = global_body_pos[:, 1:num_joints+1, :]  # (frames, 29, 3)
    else:
        # 如果数量不够，尝试只取 num_joints 个
        global_positions = global_body_pos[:, 1:1+num_joints, :] if global_body_pos.shape[1] > 1 else global_body_pos
    
    # frametime 设置为 1/60 (60fps)
    frametime = 1.0 / 60.0
    
    # 提取根关节的四元数 (qpos[:, 3:7] 是根旋转四元数，wxyz格式)
    root_quat = qpos[:, 3:7]  # (frames, 4) - 根关节四元数
    
    # 创建 NPZMotion 对象
    # qpos 只包含29个关节的角度（不包含根位置和根旋转）
    motion = NPZMotion(
        qpos=qpos_joint_angles,
        global_positions=global_positions,
        names=actual_joint_names,
        root_quat=root_quat,
        filepath=npz_path,
        frametime=frametime
    )
    
    return motion


def extract_traj(root_positions, forwards, smooth_kernel=[5, 10]):
    traj_trans, traj_angles, traj_poses = [], [], []
    FORWARD_AXIS = np.array([[0, 0, 1]]) # OpenGL system
    
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


def scaling_npz(motion, scaling_factor=0.01):
    """缩放 NPZMotion"""
    motion = motion.copy()
    motion.global_positions = motion.global_positions * scaling_factor
    return motion


def root_npz(motion, given_pos=None):
    """将根位置重置到原点（使用第一个关节的位置作为参考）"""
    motion = motion.copy()
    # 由于我们忽略了 root，使用第一个关节的位置作为参考
    root_init_pos = motion.global_positions[0, 0, [0, 2]] if given_pos is None else given_pos
    motion.global_positions[:, 0, [0, 2]] -= root_init_pos
    return motion


def on_ground_npz(motion):
    """将动作放到地面上"""
    motion = motion.copy()
    global_pos = motion.global_positions
    lowest_height = np.min(global_pos[:, :, 1])
    motion.global_positions[:, :, 1] -= lowest_height
    return motion


def mirror_npz(motion, l_joint_idxs, r_joint_idxs):
    """镜像 NPZMotion"""
    motion = motion.copy()
    ori_qpos = motion.qpos.copy()
    ori_global_positions = motion.global_positions.copy()
    
    # 镜像全局位置（X轴取反）
    motion.global_positions[:, :, 0] *= -1
    
    # 交换左右关节的全局位置
    motion.global_positions[:, l_joint_idxs] = ori_global_positions[:, r_joint_idxs]
    motion.global_positions[:, r_joint_idxs] = ori_global_positions[:, l_joint_idxs]
    # 镜像后的位置需要再次镜像X轴
    motion.global_positions[:, l_joint_idxs, 0] *= -1
    motion.global_positions[:, r_joint_idxs, 0] *= -1
    
    # 镜像 qpos：交换左右关节的角度
    # qpos 现在只包含29个关节的角度（qpos[:, 7:]），不包含根位置和根旋转
    for l_idx, r_idx in zip(l_joint_idxs, r_joint_idxs):
        if l_idx < ori_qpos.shape[1] and r_idx < ori_qpos.shape[1]:
            # 交换左右关节的角度
            motion.qpos[:, l_idx] = ori_qpos[:, r_idx]
            motion.qpos[:, r_idx] = ori_qpos[:, l_idx]
    
    # 镜像角度：取反角度值
    motion.qpos = -motion.qpos
    
    # 镜像根关节四元数
    if motion.root_quat is not None:
        ori_root_quat = motion.root_quat.copy()
        # 镜像根旋转：y和z分量取反（wxyz格式，索引1,2,3是x,y,z）
        motion.root_quat[:, [2, 3]] *= -1  # y, z分量取反
    
    return motion


def process_motion(motion, use_scale=True):
    """处理 NPZMotion"""
    if use_scale:
        motion = scaling_npz(motion, scaling_factor=0.01) # cm -> m
    motion = root_npz(motion)
    motion = on_ground_npz(motion)
    
    # 尝试使用 g1 关节名称，如果不存在则使用 mixamo 名称
    try:
        _, forwards = extract_forward_hips_npz(motion, np.arange(motion.frame_num), style100.g1_left_hip_name, style100.g1_right_hip_name, return_forward=True)
    except:
        _, forwards = extract_forward_hips_npz(motion, np.arange(motion.frame_num), style100.left_hip_name, style100.right_hip_name, return_forward=True)
    
    traj_trans, traj_angles, traj_poses = extract_traj(motion.global_positions[:, 0], forwards, smooth_kernel=[5, 10])
    
    # 直接保存 qpos（29个关节角度）和根关节四元数
    # 格式：保存原始数据，不转换为四元数
    return motion, {
        'filepath': motion.filepath,
        'local_joint_rotations': motion.qpos,  # (frames, 29) - 29个关节的角度
        'root_quat': motion.root_quat,  # (frames, 4) - 根关节四元数 (wxyz格式)
        'global_root_positions': motion.global_positions[:, 0],
        'traj': traj_trans,
        'traj_angles': traj_angles,
        'traj_pose': traj_poses
    }

if __name__ == '__main__':
    
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
        
        start_idx, end_idx = start_idx // 2, end_idx // 2
                
        motion = load_npz_motion(npz_path, start_idx=start_idx, end_idx=end_idx)
        motion, motion_data = process_motion(motion)
        motion_data['text'] = meta_info['description'].lower()
        motion_data['style'] = style_name
        data_list['motions'].append(motion_data)
        
        if meta_info['is_symmetric']:
            mirror_motion = motion.copy()
            names = mirror_motion.names
            l_names = sorted([val for val in names if 'left' in val.lower()])
            r_names = sorted([val for val in names if 'right' in val.lower()])
            l_joint_idxs, r_joint_idxs = [names.index(name) for name in l_names], [names.index(name) for name in r_names]
            mirror_motion = mirror_npz(mirror_motion, l_joint_idxs, r_joint_idxs)
            _, mirror_motion_data = process_motion(mirror_motion, use_scale=False)
            mirror_motion_data['filepath'] = mirror_motion_data['filepath'].replace('.npz', '.mirror.npz')
            mirror_motion_data['text'] = mirror_text(meta_info['description'])
            mirror_motion_data['style'] = style_name
            data_list['motions'].append(mirror_motion_data)
        print('Finish processing %s' % npz_path)
    
    # 创建 T-pose
    # T-pose 的 qpos：所有关节角度为0（29个关节）
    # T-pose 的根关节四元数：单位四元数 [1, 0, 0, 0] (wxyz格式)
    T_qpos = np.zeros((1, motion.qpos.shape[1]))  # (1, 29) - 所有角度为0
    T_root_quat = np.zeros((1, 4))  # (1, 4)
    T_root_quat[0, 0] = 1.0  # w=1，单位四元数
    T_global_positions = np.zeros((1, motion.global_positions.shape[1], motion.global_positions.shape[2]))
    T_pose_motion = NPZMotion(
        qpos=T_qpos,
        global_positions=T_global_positions,
        names=motion.names,
        root_quat=T_root_quat,
        filepath=motion.filepath,
        frametime=motion.frametime
    )
    data_list['T_pose'] = T_pose_motion
    
    pickle.dump(data_list, open(export_path, 'wb'))
    print('Finish exporting %s' % export_path)
