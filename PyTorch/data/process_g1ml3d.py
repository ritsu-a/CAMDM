import sys
sys.path.append('./')

import os
import pickle
import joblib
import numpy as np
import torch
import clip
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from utils.diff_quat import vec6d_to_quat
from data.make_pose_data import NPZMotion, convert_y_up_to_z_up


def load_g1_pickle(pickle_path):
    """
    加载G1ML3D的pickle文件
    
    注意：原始pickle文件中的数据是Y-up格式
    - Y轴（索引1）是高度轴（up）
    - Z轴（索引2）是前向轴（forward）
    - X轴（索引0）是右向轴（right）
    
    返回的字典包含：
    - angles: (frames, 29) - 29个关节的角度
    - global_translation: (frames, 3, 1) - 根位置（Y-up格式）
    - global_rotation: (frames, 3, 2) - 根旋转（6D格式，Y-up格式）
    - root_pos: (frames, 3) - 根位置（扁平化，Y-up格式）
    - root_quat: (frames, 4) - 根四元数（wxyz格式，Y-up格式）
    - fps: 帧率
    """
    data = joblib.load(pickle_path)
    
    # 转换tensor到numpy
    if isinstance(data, dict):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cpu().numpy()
    
    # 提取根位置（从(frames, 3, 1)转换为(frames, 3)）
    # 注意：这是Y-up格式的数据
    root_pos_yup = data['global_translation'].squeeze(-1)  # (frames, 3) - Y-up格式
    
    # 将6D旋转转换为四元数
    # global_rotation是(frames, 3, 2)格式，需要转换为(frames, 6)格式
    frames = data['global_rotation'].shape[0]
   
    # 转换为四元数（wxyz格式）
    # 注意：这是Y-up格式的四元数
    root_quat_yup = vec6d_to_quat(torch.tensor(data['global_rotation'])).numpy()  # (frames, 4) - xyzw格式
    # 转换为wxyz格式
    root_quat_yup = root_quat_yup[:, [3, 0, 1, 2]]  # (frames, 4) - wxyz格式，Y-up格式
    
    return {
        'angles': data['angles'],  # (frames, 29) - 关节角度不受坐标系影响
        'root_pos': root_pos_yup,  # (frames, 3) - Y-up格式
        'root_quat': root_quat_yup,  # (frames, 4) - wxyz格式，Y-up格式
        'fps': data.get('fps', 60),
        'filepath': pickle_path
    }



def process_g1_motion(pickle_path):
    """
    处理G1ML3D的pickle文件
    
    注意：原始pickle文件中的数据是Y-up格式，需要转换为Z-up格式保存
    仿照make_pose_data.py的处理方式：
    - 加载Y-up格式的数据
    - 转换为Z-up格式
    - 保存Z-up格式的数据
    
    返回处理后的motion数据字典（Z-up格式）
    """
    # 加载数据（Y-up格式）
    data = load_g1_pickle(pickle_path)
    
    angles = data['angles']  # (frames, 29) - 关节角度不受坐标系影响
    root_pos_yup = data['root_pos']  # (frames, 3) - Y-up格式
    root_quat_yup = data['root_quat']  # (frames, 4) - wxyz格式，Y-up格式
    fps = data['fps']
    frametime = 1.0 / fps
    
    # 将Y-up格式转换为Z-up格式（仿照make_pose_data.py的处理）
    # 转换根位置和根四元数
    root_pos_zup, root_quat_zup = convert_y_up_to_z_up(
        root_pos_yup.reshape(-1, 1, 3),  # 需要reshape为(frames, 1, 3)
        root_quat_yup
    )
    root_pos_zup = root_pos_zup.reshape(-1, 3)  # 转换回(frames, 3)
    
    # G1ML3D数据不需要traj数据，设置为空列表
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


def process_g1ml3d_dataset(joints_dir, texts_dir, output_path, file_list=None):
    """
    处理G1ML3D数据集
    
    joints_dir: joints文件夹路径
    texts_dir: texts文件夹路径
    output_path: 输出pkl文件路径
    file_list: 要处理的文件名列表（从train.txt或test.txt读取），如果为None则处理所有文件
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
    
    # 如果提供了文件列表，只处理列表中的文件
    if file_list is not None:
        # 将文件列表转换为集合以便快速查找
        file_set = file_list if isinstance(file_list, set) else set(file_list)
        print(f'根据文件列表处理，共 {len(file_set)} 个文件')
        
        # 获取所有可能的pickle文件名（包括带M和不带M的）
        pickle_files = []
        for file_name in file_set:
            # file_name可能是 "007440" 或 "M010416"
            # 如果file_name以M开头，直接使用；否则尝试两种格式
            if file_name.startswith('M'):
                # 文件名已经包含M前缀，直接使用
                pickle_file = f'{file_name}.pickle'
                if os.path.exists(os.path.join(joints_dir, pickle_file)):
                    pickle_files.append(pickle_file)
                else:
                    print(f'警告: 找不到文件 {file_name} 对应的pickle文件: {pickle_file}')
            else:
                # 文件名不带M前缀，尝试两种格式
                pickle_file_with_m = f'M{file_name}.pickle'
                pickle_file_without_m = f'{file_name}.pickle'
                
                if os.path.exists(os.path.join(joints_dir, pickle_file_with_m)):
                    pickle_files.append(pickle_file_with_m)
                elif os.path.exists(os.path.join(joints_dir, pickle_file_without_m)):
                    pickle_files.append(pickle_file_without_m)
                else:
                    print(f'警告: 找不到文件 {file_name} 对应的pickle文件（尝试了 {pickle_file_with_m} 和 {pickle_file_without_m}）')
        
        pickle_files = sorted(pickle_files)
    else:
        # 如果没有提供文件列表，处理所有pickle文件
        pickle_files = []
        for file in os.listdir(joints_dir):
            if file.endswith('.pickle'):
                pickle_files.append(file)
        pickle_files = sorted(pickle_files)
        print(f'未提供文件列表，处理所有 {len(pickle_files)} 个pickle文件')
    
    print(f'将处理 {len(pickle_files)} 个pickle文件')
    
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
    train_list_file = 'data/G1ML3D/train.txt'
    test_list_file = 'data/G1ML3D/test.txt'
    
    # 加载训练集和测试集文件列表
    print('=' * 60)
    print('加载训练集和测试集文件列表')
    print('=' * 60)
    train_file_list = load_file_list(train_list_file)
    test_file_list = load_file_list(test_list_file)
    
    print(f'训练集文件数: {len(train_file_list)}')
    print(f'测试集文件数: {len(test_file_list)}')
    
    # 检查是否有重叠
    overlap = train_file_list & test_file_list
    if overlap:
        print(f'警告: 训练集和测试集有 {len(overlap)} 个重叠文件: {list(overlap)[:10]}...')
    else:
        print('✓ 训练集和测试集无重叠')
    
    print('=' * 60)
    
    # 处理训练集
    print('\n处理训练集...')
    train_output_path = 'data/pkls/g1ml3d_train.pkl'
    process_g1ml3d_dataset(joints_dir, texts_dir, train_output_path, train_file_list)
    
    # 处理测试集
    print('\n处理测试集...')
    test_output_path = 'data/pkls/g1ml3d_test.pkl'
    process_g1ml3d_dataset(joints_dir, texts_dir, test_output_path, test_file_list)
    
    print('\n' + '=' * 60)
    print('处理完成！')
    print(f'训练集pkl: {train_output_path}')
    print(f'测试集pkl: {test_output_path}')
    print('=' * 60)

