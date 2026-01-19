"""
从pkl文件中读取NPZ动作数据并保存为Z-up格式的npz文件

用法: python data/export_pkl_to_npz.py [pkl_file_path] [output_dir]
"""

import sys
sys.path.append('./')

import os
import pickle
import numpy as np
from tqdm import tqdm
from data.make_pose_data import (
    NPZMotion, 
    convert_y_up_to_z_up, 
    save_npzmotion_to_npz
)


def rebuild_npzmotion_from_pkl_data(motion_data, names, try_load_from_original=True):
    """
    从pkl中的motion_data重建NPZMotion对象
    
    motion_data: pkl中保存的motion数据字典
    names: 关节名称列表
    try_load_from_original: 是否尝试从原始npz文件加载完整的global_positions
    """
    # 获取数据
    qpos = motion_data['local_joint_rotations']  # (frames, 29) - 29个关节的角度
    root_quat = motion_data.get('root_quat', None)  # (frames, 4) - 根关节四元数 (wxyz格式)
    root_pos = motion_data['global_root_positions']  # (frames, 3) - 根位置（Y-up格式）
    
    frame_num = qpos.shape[0]
    num_joints = len(names)
    
    # 尝试从原始npz文件加载完整的global_positions
    global_positions = None
    if try_load_from_original:
        original_filepath = motion_data.get('filepath', '')
        if original_filepath and os.path.exists(original_filepath):
            try:
                from data.make_pose_data import load_npz_motion
                # 加载原始文件的Y-up格式数据（与pkl中保存的格式一致）
                original_motion = load_npz_motion(original_filepath, apply_coordinate_transform=True)
                # 检查帧数是否匹配
                if original_motion.frame_num >= frame_num:
                    # 如果帧数匹配或更多，使用对应的切片
                    # 注意：这里假设pkl中的数据是原始文件的子序列
                    # 如果pkl中有framecuts信息，需要相应调整
                    global_positions = original_motion.global_positions[:frame_num]
                    print(f'  ✓ 从原始文件加载global_positions: {os.path.basename(original_filepath)}')
            except Exception as e:
                print(f'  ⚠ 无法从原始文件加载: {e}')
    
    # 如果无法从原始文件加载，创建简化的global_positions
    if global_positions is None:
        # 使用root位置作为占位符（虽然不是真实的关节位置，但保持了数据格式）
        global_positions = np.zeros((frame_num, num_joints, 3))
        # 第一个关节使用root位置
        if global_positions.shape[1] > 0:
            global_positions[:, 0, :] = root_pos
        print(f'  ⚠ 使用简化的global_positions（仅root位置）')
    
    # 创建NPZMotion对象（Y-up格式，因为pkl中保存的是Y-up格式）
    motion = NPZMotion(
        qpos=qpos,
        global_positions=global_positions,
        names=names,
        root_quat=root_quat,
        root_pos=root_pos,
        filepath=motion_data.get('filepath', ''),
        frametime=1.0 / 60.0  # 默认60fps
    )
    
    return motion


def export_pkl_to_npz(pkl_path, output_dir='data/exported_npz'):
    """
    从pkl文件读取NPZ动作数据并保存为Z-up格式的npz文件
    
    pkl_path: pkl文件路径
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'加载pkl文件: {pkl_path}')
    data_list = pickle.load(open(pkl_path, 'rb'))
    
    print(f'找到 {len(data_list["motions"])} 个motion数据')
    
    # 获取关节名称（如果有的话，否则使用默认名称）
    # 如果pkl中没有names，需要从第一个motion数据推断或使用默认值
    if data_list.get('names') is not None:
        joint_names = data_list['names']
    else:
        # 使用默认的29个关节名称（g1格式）
        # 注意：这里需要根据实际情况调整关节名称
        print('Warning: pkl中没有names信息，使用默认关节名称')
        joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
    
    # 确保joint_names数量匹配
    num_joints_in_data = data_list['motions'][0]['local_joint_rotations'].shape[1]
    if len(joint_names) != num_joints_in_data:
        print(f'Warning: 关节名称数量 ({len(joint_names)}) 与数据中的关节数量 ({num_joints_in_data}) 不匹配')
        print(f'使用前 {num_joints_in_data} 个关节名称或创建默认名称')
        if len(joint_names) > num_joints_in_data:
            joint_names = joint_names[:num_joints_in_data]
        else:
            # 补充默认名称
            for i in range(len(joint_names), num_joints_in_data):
                joint_names.append(f'joint_{i}')
    
    exported_count = 0
    skipped_count = 0
    
    for idx, motion_data in enumerate(tqdm(data_list['motions'], desc='导出NPZ文件')):
        try:
            # 重建NPZMotion对象（Y-up格式）
            # 尝试从原始npz文件加载完整的global_positions
            motion_yup = rebuild_npzmotion_from_pkl_data(motion_data, joint_names, try_load_from_original=True)
            
            # 转换为Z-up格式
            motion_zup = motion_yup.copy()
            
            # 转换全局位置
            motion_zup.global_positions = convert_y_up_to_z_up(motion_yup.global_positions)
            
            # 转换根位置和根四元数
            if motion_yup.root_pos is not None:
                motion_zup.root_pos, motion_zup.root_quat = convert_y_up_to_z_up(
                    motion_yup.root_pos.reshape(-1, 1, 3),
                    motion_yup.root_quat
                )
                motion_zup.root_pos = motion_zup.root_pos.reshape(-1, 3)
            elif motion_yup.root_quat is not None:
                root_pos_yup = motion_yup.global_positions[:, 0:1, :]
                _, motion_zup.root_quat = convert_y_up_to_z_up(root_pos_yup, motion_yup.root_quat)
                motion_zup.root_pos = motion_zup.global_positions[:, 0, :]
            
            # 生成输出文件名
            filepath = motion_data.get('filepath', '')
            if filepath:
                # 从原始文件路径提取文件名
                basename = os.path.basename(filepath)
                if basename.endswith('.npz'):
                    output_filename = basename
                else:
                    # 如果没有原始文件名，使用索引
                    output_filename = f'motion_{idx:04d}.npz'
            else:
                # 使用索引和style信息
                style = motion_data.get('style', 'unknown')
                output_filename = f'{style}_{idx:04d}.npz'
            
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存为npz文件（不需要original_data，因为没有）
            save_npzmotion_to_npz(motion_zup, output_path, original_npz_data=None)
            exported_count += 1
            
        except Exception as e:
            print(f'\nError processing motion {idx}: {e}')
            skipped_count += 1
            continue
    
    print(f'\n导出完成!')
    print(f'  成功导出: {exported_count} 个文件')
    if skipped_count > 0:
        print(f'  跳过: {skipped_count} 个文件')
    print(f'  输出目录: {output_dir}')


if __name__ == '__main__':
    import sys
    
    # 默认路径
    default_pkl_path = 'data/pkls/aistpp_g1.pkl'
    default_output_dir = 'data/exported_npz'
    
    # 从命令行参数获取路径
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    else:
        pkl_path = default_pkl_path
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = default_output_dir
    
    # 检查文件是否存在
    if not os.path.exists(pkl_path):
        print(f'错误: 找不到pkl文件: {pkl_path}')
        print(f'用法: python data/export_pkl_to_npz.py [pkl_file_path] [output_dir]')
        sys.exit(1)
    
    print('=' * 60)
    print('从PKL导出NPZ文件')
    print('=' * 60)
    print(f'输入pkl文件: {pkl_path}')
    print(f'输出目录: {output_dir}')
    print('=' * 60)
    
    try:
        export_pkl_to_npz(pkl_path, output_dir)
    except Exception as e:
        print(f'导出失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

