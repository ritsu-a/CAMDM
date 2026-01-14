import sys
sys.path.append('./')

import os
import pickle
import joblib
import numpy as np
import torch
import clip
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from utils.diff_quat import vec6d_to_quat
from data.make_pose_data import extract_traj, NPZMotion


def load_g1_pickle(pickle_path):
    """
    加载G1ML3D的pickle文件
    
    返回的字典包含：
    - angles: (frames, 29) - 29个关节的角度
    - global_translation: (frames, 3, 1) - 根位置
    - global_rotation: (frames, 3, 2) - 根旋转（6D格式）
    - root_pos: (frames, 3) - 根位置（扁平化）
    - root_quat: (frames, 4) - 根四元数（wxyz格式）
    - fps: 帧率
    """
    data = joblib.load(pickle_path)
    
    # 转换tensor到numpy
    if isinstance(data, dict):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cpu().numpy()
    
    # 提取根位置（从(frames, 3, 1)转换为(frames, 3)）
    root_pos = data['global_translation'].squeeze(-1)  # (frames, 3)
    
    # 将6D旋转转换为四元数
    # global_rotation是(frames, 3, 2)格式，需要转换为(frames, 6)格式
    frames = data['global_rotation'].shape[0]
   
    # 转换为四元数（wxyz格式）
    root_quat = vec6d_to_quat(torch.tensor(data['global_rotation'])).numpy()  # (frames, 4) - xyzw格式
    # 转换为wxyz格式
    root_quat = root_quat[:, [3, 0, 1, 2]]  # (frames, 4) - wxyz格式
    
    return {
        'angles': data['angles'],  # (frames, 29)
        'root_pos': root_pos,  # (frames, 3)
        'root_quat': root_quat,  # (frames, 4) - wxyz格式
        'fps': data.get('fps', 60),
        'filepath': pickle_path
    }


def extract_forward_from_root(root_quat):
    """
    从根四元数提取前向方向
    
    root_quat: (frames, 4) - wxyz格式
    返回: (frames, 3) - 前向方向向量
    """
    # 转换为xyzw格式用于Rotation
    quat_xyzw = root_quat[:, [1, 2, 3, 0]]  # (frames, 4) - xyzw格式
    
    # 默认前向方向是Z轴（在Y-up系统中，Z是前向）
    default_forward = np.array([[0, 0, 1]])  # Y-up system: Z-forward
    
    forwards = []
    for i in range(root_quat.shape[0]):
        rot = R.from_quat(quat_xyzw[i])
        forward = rot.apply(default_forward)  # 旋转默认前向方向
        forwards.append(forward[0])
    
    return np.array(forwards)  # (frames, 3)




def process_g1_motion(pickle_path):
    """
    处理G1ML3D的pickle文件
    
    返回处理后的motion数据字典
    """
    # 加载数据
    data = load_g1_pickle(pickle_path)
    
    angles = data['angles']  # (frames, 29)
    root_pos = data['root_pos']  # (frames, 3)
    root_quat = data['root_quat']  # (frames, 4) - wxyz格式
    fps = data['fps']
    frametime = 1.0 / fps
    
    
    # 从根四元数提取前向方向
    forwards = extract_forward_from_root(root_quat)  # (frames, 3)
    
    # 提取轨迹
    traj_trans, traj_angles, traj_poses = extract_traj(root_pos, forwards, smooth_kernel=[5, 10])
    
    return {
        'filepath': pickle_path,
        'local_joint_rotations': angles,  # (frames, 29)
        'root_quat': root_quat,  # (frames, 4) - wxyz格式
        'global_root_positions': root_pos,  # (frames, 3)
        'traj': traj_trans,
        'traj_angles': traj_angles,
        'traj_pose': traj_poses
    }


def load_texts_from_file(text_path):
    """
    从文本文件加载所有可能的文本描述
    
    文本文件格式：每行是一个描述，用#分隔，第一部分是文本描述
    
    返回: 文本列表
    """
    texts = []
    if not os.path.exists(text_path):
        return texts
    
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 提取#之前的部分作为文本描述
            parts = line.split('#')
            if len(parts) > 0:
                text = parts[0].strip()
                if text:
                    texts.append(text)
    
    return texts


def process_g1ml3d_dataset(joints_dir, texts_dir, output_path):
    """
    处理G1ML3D数据集
    
    joints_dir: joints文件夹路径
    texts_dir: texts文件夹路径
    output_path: 输出pkl文件路径
    """
    # 加载CLIP模型用于文本编码
    print('加载CLIP模型用于文本特征提取...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    print(f'CLIP模型已加载到 {device}')
    
    data_list = {
        'parents': None,
        'offsets': None,
        'names': None,
        'motions': []
    }
    
    # 获取所有pickle文件
    pickle_files = []
    for file in os.listdir(joints_dir):
        if file.endswith('.pickle'):
            pickle_files.append(file)
    pickle_files = sorted(pickle_files)
    
    print(f'找到 {len(pickle_files)} 个pickle文件')
    
    for pickle_file in tqdm(pickle_files):
        pickle_path = os.path.join(joints_dir, pickle_file)
        
        # 找到对应的文本文件
        # 文件名可能是 "000000.pickle" 或 "M000000.pickle"，对应的文本文件是 "000000.txt"
        base_name = os.path.splitext(pickle_file)[0]
        # 如果以M开头，去掉M
        if base_name.startswith('M'):
            base_name = base_name[1:]
        text_file = f'{base_name}.txt'
        text_path = os.path.join(texts_dir, text_file)
        
        # 加载所有可能的文本描述
        texts = load_texts_from_file(text_path)
        
        if not texts:
            print(f'警告: 未找到文本文件或文本为空: {text_path}，跳过 {pickle_file}')
            continue
        
        # 处理动作数据
        try:
            motion_data = process_g1_motion(pickle_path)
        except Exception as e:
            print(f'错误: 处理 {pickle_file} 时出错: {e}，跳过')
            continue
       
        
        # 为每个文本描述提取CLIP特征
        text_features_list = []
        texts_lower = []
        
        for text in texts:
            text_lower = text.lower()
            texts_lower.append(text_lower)
            
            # 计算CLIP文本特征
            text_tokens = clip.tokenize([text_lower], truncate=True).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tokens).float().cpu().numpy()  # (1, 512)
            text_features_list.append(text_features[0])  # (512,) numpy array
        
        # 保存所有文本特征
        motion_data['texts'] = texts_lower  # 所有可能的文本描述
        motion_data['text_features'] = np.array(text_features_list)  # (num_texts, 512)
        # 为了兼容性，也保存第一个文本作为默认文本
        motion_data['text'] = texts_lower[0]
        motion_data['text_feature'] = text_features_list[0]  # (512,)
        # 添加style字段（使用文件名作为style）
        motion_data['style'] = base_name
        
        data_list['motions'].append(motion_data)
    
    # 保存数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pickle.dump(data_list, open(output_path, 'wb'))
    print(f'处理完成，已保存到: {output_path}')
    print(f'总共处理了 {len(data_list["motions"])} 个动作')


if __name__ == '__main__':
    # 设置路径
    joints_dir = 'data/G1ML3D/joints'
    texts_dir = 'data/G1ML3D/texts'
    output_path = 'data/pkls/g1ml3d.pkl'
    
    # 处理数据集
    process_g1ml3d_dataset(joints_dir, texts_dir, output_path)

