import sys
sys.path.append('./')
import numpy as np
if np.__version__[:2] == "1.":
    import sys
    # More comprehensive compatibility mapping for NumPy 2.x pickles
    try:
        sys.modules["numpy._core.numeric"] = np.core.numeric
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
        sys.modules["numpy._core.umath"] = np.core.umath
        sys.modules["numpy._core.arrayprint"] = np.core.arrayprint
        sys.modules["numpy._core.fromnumeric"] = np.core.fromnumeric
        sys.modules["numpy._core.defchararray"] = np.core.defchararray
        sys.modules["numpy._core.records"] = np.core.records
        sys.modules["numpy._core.function_base"] = np.core.function_base
        sys.modules["numpy._core.machar"] = np.core.machar
        sys.modules["numpy._core.getlimits"] = np.core.getlimits
        sys.modules["numpy._core.shape_base"] = np.core.shape_base
        sys.modules["numpy._core.stride_tricks"] = np.core.stride_tricks
        sys.modules["numpy._core.einsumfunc"] = np.core.einsumfunc
        sys.modules["numpy._core._asarray"] = np.core._asarray
        sys.modules["numpy._core._dtype_ctypes"] = np.core._dtype_ctypes
        sys.modules["numpy._core._internal"] = np.core._internal
        sys.modules["numpy._core._dtype"] = np.core._dtype
        sys.modules["numpy._core._exceptions"] = np.core._exceptions
        sys.modules["numpy._core._methods"] = np.core._methods
        sys.modules["numpy._core._type_aliases"] = np.core._type_aliases
        sys.modules["numpy._core._ufunc_config"] = np.core._ufunc_config
        sys.modules["numpy._core._add_newdocs"] = np.core._add_newdocs
        sys.modules["numpy._core._add_newdocs_scalars"] = np.core._add_newdocs_scalars
        sys.modules["numpy._core._multiarray_tests"] = np.core._multiarray_tests
        sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath
        sys.modules["numpy._core._operand_flag_tests"] = np.core._operand_flag_tests
        sys.modules["numpy._core._struct_ufunc_tests"] = np.core._struct_ufunc_tests
        sys.modules["numpy._core._umath_tests"] = np.core._umath_tests
    except AttributeError:
        # Some modules might not exist in older NumPy versions
        pass

import os
import pickle
from tqdm import tqdm
from scipy import signal
from scipy.spatial.transform import Rotation as R


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



def load_aist_motion(pickle_path):
    """
    加载AIST的motion pickle文件
    
    注意：原始pickle文件中的数据格式：
    - root_pos: (frames, 3) - 根位置
    - root_rot: (frames, 4) - 根旋转四元数（wxyz格式）
    - dof_pos: (frames, 29) - 29个关节的角度
    - fps: 帧率
    
    返回的字典包含：
    - angles: (frames, 29) - 29个关节的角度
    - root_pos: (frames, 3) - 根位置
    - root_quat: (frames, 4) - 根四元数（wxyz格式）
    - fps: 帧率
    """
    data = pickle.load(open(pickle_path, 'rb'))
    
    # 提取数据
    angles = data['dof_pos']  # (frames, 29) - 关节角度
    root_pos = data['root_pos']  # (frames, 3) - 根位置
    root_quat = data['root_rot'][:, [3, 0, 1, 2]]  # (frames, 4) - 根四元数（wxyz格式）
    fps = data['fps']
    
    return {
        'angles': angles,
        'root_pos': root_pos,
        'root_quat': root_quat,
        'fps': fps,
        'filepath': pickle_path
    }


def process_aist_motion(pickle_path):
    """
    处理AIST的motion pickle文件
    
    注意：假设原始数据是Y-up格式，需要转换为Z-up格式保存
    仿照process_g1ml3d.py的处理方式：
    - 加载Y-up格式的数据
    - 应用低通滤波器
    - 转换为Z-up格式
    - 保存Z-up格式的数据
    
    返回处理后的motion数据字典（Z-up格式）
    """
    # 加载数据（Y-up格式）
    data = load_aist_motion(pickle_path)
    
    angles = data['angles']  # (frames, 29) - 关节角度不受坐标系影响
    root_pos_yup = data['root_pos']  # (frames, 3) - Y-up格式
    root_quat_yup = data['root_quat']  # (frames, 4) - wxyz格式，Y-up格式
    fps = data['fps']
    frametime = 1.0 / fps
  

    # 重新归一化四元数以保持单位约束
    quat_norm = np.linalg.norm(root_quat_yup, axis=1, keepdims=True)
    root_quat_yup = root_quat_yup / (quat_norm + 1e-8)
    
    # 将Y-up格式转换为Z-up格式（仿照make_pose_data.py的处理）
    # 转换根位置和根四元数
    root_pos_zup, root_quat_zup = convert_y_up_to_z_up(
        root_pos_yup.reshape(-1, 1, 3),  # 需要reshape为(frames, 1, 3)
        root_quat_yup
    )
    root_pos_zup = root_pos_zup.reshape(-1, 3)  # 转换回(frames, 3)
    
    # AIST数据不需要traj数据，设置为空列表
    # 保持与extract_traj返回格式一致（列表格式，对应smooth_kernel=[5, 10]）
    traj_trans = []  # 空列表
    traj_angles = []  # 空列表
    traj_poses = []  # 空列表
    
    return {
        'filepath': pickle_path,
        'local_joint_rotations': angles,  # (frames, 29) - 关节角度不受坐标系影响
        'root_quat': root_quat_zup,  # (frames, 4) - wxyz格式，Z-up格式
        'global_root_positions': root_pos_zup,  # (frames, 3) - Z-up格式
        'traj': traj_trans,
        'traj_angles': traj_angles,
        'traj_pose': traj_poses
    }


def load_audio_feature(audio_path):
    """
    加载音频特征文件（npy格式）
    
    参数:
        audio_path: 音频特征文件路径
        
    返回:
        numpy.ndarray: 音频特征 (frames, feature_dim)，如果文件损坏或不存在则返回None
    """
    if not os.path.exists(audio_path):
        return None
    try:
        return np.load(audio_path, allow_pickle=True)
    except (EOFError, pickle.UnpicklingError, ValueError, OSError) as e:
        print(f'警告: 无法加载音频特征文件 {audio_path}: {e}，跳过')
        return None


def get_motion_base_name(motion_filename):
    """
    从motion文件名提取基础名称（用于匹配音频特征文件）
    
    例如: gBR_sBM_cAll_d04_mBR1_ch03_slice0_rtg.pkl -> gBR_sBM_cAll_d04_mBR1_ch03_slice0
    """
    base_name = os.path.splitext(motion_filename)[0]  # 去掉 .pkl
    # 去掉 _rtg 后缀（如果存在）
    if base_name.endswith('_rtg'):
        base_name = base_name[:-4]
    return base_name


def process_aist_dataset(motions_dir, audio_feats_dir, output_path, file_list=None):
    """
    处理AIST数据集
    
    参数:
        motions_dir: motions文件夹路径
        audio_feats_dir: 音频特征文件夹路径
        output_path: 输出pkl文件路径
        file_list: 要处理的文件名列表（从name.txt读取），如果为None则处理所有文件
    """
    data_list = {
        'parents': None,
        'offsets': None,
        'names': None,
        'motions': []
    }
    
    # 如果提供了文件列表，只处理列表中的文件
    if file_list is not None:
        # 将文件列表转换为集合以便快速查找
        file_set = file_list if isinstance(file_list, set) else set(file_list)
        print(f'根据文件列表处理，共 {len(file_set)} 个文件')
        
        # 获取所有可能的motion文件名
        motion_files = []
        for file_name in file_set:
            # file_name可能是 "gBR_sBM_cAll_d04_mBR1_ch03_slice0_rtg.pkl" 或 "gBR_sBM_cAll_d04_mBR1_ch03_slice0"
            # 确保文件名以 .pkl 结尾
            if not file_name.endswith('.pkl'):
                file_name = file_name + '.pkl'
            
            motion_file = file_name
            if os.path.exists(os.path.join(motions_dir, motion_file)):
                motion_files.append(motion_file)
            else:
                print(f'警告: 找不到motion文件: {motion_file}')
        
        motion_files = sorted(motion_files)
    else:
        # 如果没有提供文件列表，处理所有pkl文件
        motion_files = []
        for file in os.listdir(motions_dir):
            if file.endswith('.pkl'):
                motion_files.append(file)
        motion_files = sorted(motion_files)
        print(f'未提供文件列表，处理所有 {len(motion_files)} 个motion文件')
    
    print(f'将处理 {len(motion_files)} 个motion文件')
    
    for motion_file in tqdm(motion_files):
        motion_path = os.path.join(motions_dir, motion_file)
        
        # 找到对应的音频特征文件
        base_name = get_motion_base_name(motion_file)
        audio_file = f'{base_name}.npy'
        audio_path = os.path.join(audio_feats_dir, audio_file)
        
        # 加载音频特征
        audio_feature = load_audio_feature(audio_path)
        
        if audio_feature is None:
            print(f'警告: 未找到音频特征文件: {audio_path}，跳过 {motion_file}')
            continue
        
        # 处理动作数据
        try:
            motion_data = process_aist_motion(motion_path)
        except Exception as e:
            print(f'错误: 处理 {motion_file} 时出错: {e}，跳过')
            continue
        
        # 检查音频特征和motion数据的帧数是否匹配
        motion_frames = motion_data['local_joint_rotations'].shape[0]
        audio_frames = audio_feature.shape[0]
        
        if motion_frames != audio_frames:
            print(f'警告: {motion_file} 的帧数 ({motion_frames}) 与音频特征帧数 ({audio_frames}) 不匹配，跳过')
            continue
        
        # 保存音频特征
        motion_data['audio_feature'] = audio_feature  # (frames, 4800)
        # 为了兼容性，也保存第一个帧的音频特征作为默认特征
        motion_data['audio_feat'] = audio_feature[0]  # (4800,)
        # 添加style字段（使用文件名作为style）
        motion_data['style'] = base_name
        
        data_list['motions'].append(motion_data)
    
    # 保存数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pickle.dump(data_list, open(output_path, 'wb'))
    print(f'处理完成，已保存到: {output_path}')
    print(f'总共处理了 {len(data_list["motions"])} 个动作')


def load_file_list(list_file_path):
    """
    从文件中加载文件名列表
    
    返回: 文件名集合（去除换行符和空行）
    """
    file_set = set()
    if not os.path.exists(list_file_path):
        print(f'警告: 文件列表不存在: {list_file_path}')
        return file_set
    
    with open(list_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                file_set.add(line)
    
    return file_set


if __name__ == '__main__':
    # 设置路径
    motions_dir = 'data/aistpp_g1/motions_g1'
    audio_feats_dir = 'data/aistpp_g1/jukebox_feats'
    name_file = 'data/aistpp_g1/name.txt'
    
    # 加载文件名列表
    print('=' * 60)
    print('加载文件名列表')
    print('=' * 60)
    file_list = load_file_list(name_file)
    print(f'文件总数: {len(file_list)}')
    print('=' * 60)
    
    # 处理数据集（不划分训练集和测试集，全部处理）
    print('\n处理AIST数据集...')
    output_path = 'data/pkls/aistpp_g1.pkl'
    process_aist_dataset(motions_dir, audio_feats_dir, output_path, file_list)
    
    print('\n' + '=' * 60)
    print('处理完成！')
    print(f'输出pkl: {output_path}')
    print('=' * 60)

