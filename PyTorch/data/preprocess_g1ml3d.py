import sys

import numpy as np

sys.modules["numpy._core.numeric"] = np.core.numeric
sys.modules["numpy._core.multiarray"] = np.core.multiarray
import os as _os

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PYTORCH_ROOT = _os.path.abspath(_os.path.join(_THIS_DIR, '..'))
if _PYTORCH_ROOT not in sys.path:
    sys.path.insert(0, _PYTORCH_ROOT)

import os
import glob
import pickle

from tqdm import tqdm
import torch
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from utils.g1_motion import G1Motion
from utils.nn_transforms import repr6d2quat
from utils.diff_quat import vec6d_to_quat


def text_feat_path_from_motion_path(motion_pkl_path: str, texts_dir: str) -> str:
    """
    motion: .../joints/XXXX.pickle
    text  : .../texts/XXXX.txt
    """
    base = os.path.basename(motion_pkl_path)
    if not base.endswith('.pickle'):
        raise ValueError(f'Unexpected motion filename: {base}')
    text_name = base.replace('.pickle', '.txt')
    return os.path.join(texts_dir, text_name)


def load_text_description(text_path: str) -> str:
    """
    从文本文件中读取描述。
    文本文件格式：每行一个描述，用 # 分隔，第一部分是纯文本描述。
    返回第一行的第一部分（纯文本描述）。
    """
    if not os.path.exists(text_path):
        raise RuntimeError(f'Missing text file: {text_path}')
    
    with open(text_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) == 0:
            return ''
        # 取第一行，用 # 分隔，取第一部分
        first_line = lines[0].strip()
        if '#' in first_line:
            return first_line.split('#')[0].strip()
        return first_line


def encode_text_with_clip(text: str, clip_model, device='cuda'):
    """
    使用 CLIP 模型编码文本。
    返回文本特征向量 (512,)，CLIP ViT-B/32 的特征维度。
    """
    import clip
    
    if not text:
        # 如果文本为空，返回零向量
        return np.zeros(512, dtype=np.float32)
    
    # 使用 CLIP 的 tokenize 函数
    text_tokens = clip.tokenize([text], truncate=True).to(device)
    
    # 编码文本
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        # 归一化特征
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy().astype(np.float32)
    
    return text_features[0]  # 返回 (512,) 的特征向量


def apply_lowpass_filter(data: np.ndarray,  cutoff_freq: float = 0.2, order: int = 4) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter to the data.
    
    Parameters:
        data (numpy.ndarray): The input data to be filtered.
        cutoff_freq (float): The normalized cutoff frequency (0~1).
        order (int): The order of the Butterworth filter.
        
    Returns:
        numpy.ndarray: The filtered data.
    """
    b, a = signal.butter(order, cutoff_freq, 'low')
    filtered_data = data.copy()
    for idx in range(filtered_data.shape[1]):
        filtered_data[:, idx] = signal.filtfilt(b, a, data[:, idx])
    return filtered_data


def apply_gaussian_smooth(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    使用高斯滤波器平滑数据（替代方案）。
    
    Parameters:
        data: (T, D) 输入数据
        sigma: 高斯核标准差，默认 1.0
    
    Returns:
        平滑后的数据 (T, D)
    """
    if data.shape[0] < 3:
        return data
    
    filtered_data = data.copy()
    for dim in range(data.shape[1]):
        filtered_data[:, dim] = gaussian_filter1d(data[:, dim], sigma=sigma, mode='nearest')
    
    return filtered_data







def expand_text_feat_to_frames(text_feat: np.ndarray, num_frames: int) -> np.ndarray:
    """
    将文本特征扩展到与帧数匹配。
    text_feat: (feat_dim,) - CLIP 文本特征，通常是 512 维
    返回: (num_frames, feat_dim) - 每帧使用相同的文本特征
    """
    if len(text_feat.shape) != 1:
        raise ValueError(f'Expected 1D text feature, got shape {text_feat.shape}')
    
    feat_dim = text_feat.shape[0]
    
    # 将文本特征重复到每一帧
    text_feat_expanded = np.repeat(text_feat[np.newaxis, :], num_frames, axis=0)
    
    return text_feat_expanded.astype(np.float32)


def load_split_file(split_path: str) -> set:
    """
    加载 train.txt 或 test.txt 文件，返回文件名集合（不带扩展名）。
    """
    if not os.path.exists(split_path):
        raise RuntimeError(f'Split file not found: {split_path}')
    
    file_names = set()
    with open(split_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:  # 忽略空行
                file_names.add(name)
    
    return file_names


def process_motion_data(pkl_path: str, texts_dir: str, clip_model, device: str, dof_dim: int = 36, 
                        apply_filter: bool = True, filter_cutoff_freq: float = 5.0):
    """
    处理单个运动数据文件。
    返回 motion_data 字典，如果处理失败返回 None。
    
    Parameters:
        apply_filter: 是否应用低通滤波器，默认 True
        filter_cutoff_freq: 滤波器截止频率（Hz），默认 5.0 Hz
    """
    try:
        raw = pickle.load(open(pkl_path, 'rb'))
        fps = float(raw.get('fps', 20.0))

        # 提取数据
        angles = np.asarray(raw['angles'], dtype=np.float32)  # (T, 29)
        global_rotation = np.asarray(raw['global_rotation'], dtype=np.float32)  # (T, 3, 2)
        global_translation = np.asarray(raw['global_translation'], dtype=np.float32)  # (T, 3, 1)

        # 转换 global_translation: (T, 3, 1) -> (T, 3)
        root_pos = global_translation[:, :, 0]  # (T, 3)

        # 转换 global_rotation 为四元数 (wxyz 格式)
        # vec6d_to_quat 需要 torch.Tensor，输入格式为 (..., 3, 2)
        rot_6d_torch = torch.from_numpy(global_rotation).float()  # (T, 3, 2)
        quat_torch = vec6d_to_quat(rot_6d_torch)  # (T, 4) xyzw
        root_rot_wxyz = quat_torch.numpy().astype(np.float32)[:, [3, 0, 1, 2]]

        
        angles = apply_lowpass_filter(angles)
        # 组合成 dof: root_pos + root_rot + angles
        dof = np.concatenate([root_pos, root_rot_wxyz, angles], axis=-1)  # (T, 36)
        
        if dof.shape[-1] != dof_dim:
            raise RuntimeError(f'Unexpected dof dim: got {dof.shape[-1]} expected {dof_dim} for {pkl_path}')

        motion = G1Motion(dof_pos=dof, fps=fps, filepath=pkl_path)

        # 读取文本文件
        text_path = text_feat_path_from_motion_path(pkl_path, texts_dir)
        if not os.path.exists(text_path):
            return None, f'Missing text file: {text_path}'

        # 读取文本描述
        text_description = load_text_description(text_path)
        
        # 使用 CLIP 编码文本
        try:
            text_feat = encode_text_with_clip(text_description, clip_model, device=device)
            # 扩展文本特征到与帧数匹配
            text_feat_expanded = expand_text_feat_to_frames(text_feat, motion.frame_num)
        except Exception as e:
            return None, f'CLIP encoding error: {e}'

        # 验证帧数匹配
        assert text_feat_expanded.shape[0] == motion.frame_num, \
            f'Frame mismatch: motion={motion.frame_num} text_feat={text_feat_expanded.shape[0]}'

        motion_data = {
            'filepath': pkl_path,
            'fps': fps,
            'dof_pos': motion.dof_pos,
            'audio_feat': text_feat_expanded,  # 使用 text_feat 代替 audio_feat
            'text': text_description,
        }
        return motion_data, None

    except Exception as e:
        return None, f'Error processing {pkl_path}: {e}'


if __name__ == '__main__':
    joints_dir = '/root/workspace/CAMDM/PyTorch/data/G1ML3D_v1/joints'
    texts_dir = '/root/workspace/CAMDM/PyTorch/data/G1ML3D_v1/texts'
    train_split_file = '/root/workspace/CAMDM/PyTorch/data/G1ML3D_v1/train.txt'
    test_split_file = '/root/workspace/CAMDM/PyTorch/data/G1ML3D_v1/test.txt'
    export_train_path = '/root/workspace/CAMDM/PyTorch/data/pkls/g1ml3d_v1_train.pkl'
    export_test_path = '/root/workspace/CAMDM/PyTorch/data/pkls/g1ml3d_v1_test.pkl'
    
    os.makedirs(os.path.dirname(export_train_path), exist_ok=True)

    # 加载 CLIP 模型
    try:
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Loading CLIP model on {device}...')
        clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
        clip_model.eval()
        # CLIP 的 tokenizer 是通过 clip.tokenize 函数使用的
        print('CLIP model loaded successfully.')
    except ImportError:
        raise RuntimeError('CLIP library not found. Please install: pip install clip-by-openai')
    except Exception as e:
        raise RuntimeError(f'Failed to load CLIP model: {e}')

    # 加载 train 和 test split
    print('Loading train/test splits...')
    train_names = load_split_file(train_split_file)
    test_names = load_split_file(test_split_file)
    print(f'Loaded {len(train_names)} train samples, {len(test_names)} test samples')

    if not os.path.isdir(texts_dir):
        raise RuntimeError(f'texts folder not found: {texts_dir}')

    # (3 root pos + 4 root rot(wxyz) + 29 dof) = 36
    dof_dim = 36

    def process_split(split_names: set, split_name: str):
        """处理一个 split（train 或 test）"""
        data_list = {
            'motions': []
        }
        skipped = 0
        
        # 获取所有 pickle 文件
        all_pkl_files = sorted(glob.glob(os.path.join(joints_dir, '*.pickle')))
        
        # 根据 split_names 过滤文件
        pkl_files = []
        for pkl_path in all_pkl_files:
            base_name = os.path.basename(pkl_path).replace('.pickle', '')
            if base_name in split_names:
                pkl_files.append(pkl_path)
        
        print(f'\nProcessing {split_name} split: {len(pkl_files)} files')
        
        for pkl_path in tqdm(pkl_files, desc=f'Processing {split_name}'):
            motion_data, error_msg = process_motion_data(
                pkl_path, texts_dir, clip_model, device, dof_dim,
                apply_filter=True, filter_cutoff_freq=5.0  # 应用低通滤波器，截止频率 5 Hz
            )
            
            if motion_data is None:
                skipped += 1
                print(f'[WARN] skip {os.path.basename(pkl_path)}: {error_msg}')
                continue
            
            data_list['motions'].append(motion_data)
        
        return data_list, skipped

    # 处理训练集
    train_data_list, train_skipped = process_split(train_names, 'train')
    pickle.dump(train_data_list, open(export_train_path, 'wb'))
    print(f'\nFinish exporting {export_train_path} (motions={len(train_data_list["motions"])}, skipped={train_skipped})')

    # 处理测试集
    test_data_list, test_skipped = process_split(test_names, 'test')
    pickle.dump(test_data_list, open(export_test_path, 'wb'))
    print(f'Finish exporting {export_test_path} (motions={len(test_data_list["motions"])}, skipped={test_skipped})')
    
    print(f'\nTotal: train={len(train_data_list["motions"])}, test={len(test_data_list["motions"])}, '
          f'train_skipped={train_skipped}, test_skipped={test_skipped}')

