import sys
sys.path.append('./')

import torch
import pickle
import random
import numpy as np

import utils.nn_transforms as nn_transforms
from scipy.ndimage import gaussian_filter1d

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torch.utils.data import Dataset

# 导入NPZMotion类以支持pickle加载
try:
    from data.make_pose_data import NPZMotion
    # 将NPZMotion注册到__main__模块，以便pickle能够找到它
    import __main__
    if not hasattr(__main__, 'NPZMotion'):
        __main__.NPZMotion = NPZMotion
except ImportError:
    # 如果导入失败，定义占位类
    class NPZMotion:
        pass
    import __main__
    if not hasattr(__main__, 'NPZMotion'):
        __main__.NPZMotion = NPZMotion


class MotionDataset(Dataset):
    '''
    rot_req: str, rotation format, 'q'|'6d'|'euler'
    window_size: int, window size for each clip
    '''
    def __init__(self, pkl_path, rot_req, offset_frame, past_frame, future_frame, dtype=np.float32, limited_num=-1):
        self.pkl_path, self.rot_req,self.dtype = pkl_path, rot_req, dtype
        window_size = past_frame + future_frame
        self.past_frame = past_frame
        self.rotations_list, self.root_pos_list = [], []
        self.local_conds = {'traj_pose': [], 'traj_trans': []}
        self.global_conds = {'style': [], 'text': [], 'text_feature': []}
        
        self.rot_feat_dim = {'q': 4, '6d': 6, 'euler': 3, 'qpos': 1}
        self.reference_frame_idx = past_frame
        
        data_source = pickle.load(open(pkl_path, 'rb'))

        frame_nums = []
        item_frame_indices_list = [] # store the frame index for each clip. shape = 1 + window_size, 1 represent the motion idx
        motion_idx = 0
        for motion_item in tqdm(data_source['motions'][:limited_num]):
            frame_num = motion_item['local_joint_rotations'].shape[0]
            if frame_num < window_size:
                continue
            frame_nums.append([motion_item['style'], frame_num])
            
            # 直接保存qpos，不转换为旋转
            # local_joint_rotations 是 qpos (frames, 29) - 29个关节的角度
            local_joint_rotations = motion_item['local_joint_rotations'].astype(dtype)
            
            # 保存qpos和root_quat的原始格式
            qpos_data = {
                'qpos': local_joint_rotations,  # (frames, 29) - 29个关节的角度
                'root_quat': motion_item.get('root_quat', None)  # (frames, 4) - 根关节四元数，可选
            }
            
            # 如果数据是2D，说明是qpos格式，直接保存
            if len(local_joint_rotations.shape) == 2:
                self.rotations_list.append(qpos_data)
            else:
                # 如果已经是其他格式（向后兼容），直接保存
                self.rotations_list.append(local_joint_rotations)
            
            self.root_pos_list.append(motion_item['global_root_positions'].astype(dtype))
            
            self.local_conds['traj_pose'].append(np.array([item for item in motion_item['traj_pose']], dtype=dtype))
            self.local_conds['traj_trans'].append(np.array([item for item in motion_item['traj']], dtype=dtype))
            
            self.global_conds['style'].append(motion_item['style'])
            # 保存文本信息，如果不存在则使用style作为文本
            text = motion_item.get('text', motion_item['style'])
            self.global_conds['text'].append(text)
            # 加载预计算的CLIP特征，如果不存在则报错（需要先运行make_pose_data.py预处理）
            if 'text_feature' in motion_item:
                self.global_conds['text_feature'].append(motion_item['text_feature'].astype(np.float32))
            else:
                raise ValueError(f"text_feature not found in motion_item. Please run make_pose_data.py to precompute CLIP features.")

            clip_indices = np.arange(0, frame_num - window_size + 1, offset_frame)[:, None] + np.arange(window_size)
            clip_indices_with_idx = np.hstack((np.full((len(clip_indices), 1), motion_idx, dtype=clip_indices.dtype), clip_indices))            
            item_frame_indices_list.append(clip_indices_with_idx)
            motion_idx += 1
            
        self.item_frame_indices = np.concatenate(item_frame_indices_list, axis=0)
    
        # 确定关节数量和特征维度
        first_rot_data = self.rotations_list[0]
        if isinstance(first_rot_data, dict):
            # qpos格式：29个关节 + 1个根关节 = 30
            self.joint_num = 30
        else:
            self.joint_num = first_rot_data.shape[-2]
        self.per_rot_feat = self.rot_feat_dim[rot_req]
        self.traj_aug_indexs1 = list(range(self.local_conds['traj_pose'][0].shape[0]))
        self.traj_aug_indexs2 = list(range(self.local_conds['traj_trans'][0].shape[0]))
        self.mask = np.ones(window_size - self.reference_frame_idx, dtype=bool)
        self.style_set = sorted(list(set(self.global_conds['style'])))
        print('Dataset loaded, trained with %d clips, %d frames, %d mins in total' % (len(frame_nums), sum([item[1] for item in frame_nums]), sum([item[1] for item in frame_nums])/30/60))
        print('Using precomputed CLIP text features instead of style idx')
        
        
    def __len__(self):
        return len(self.item_frame_indices)
    
    def __getitem__(self, idx):
        item_frame_indice = self.item_frame_indices[idx]
        motion_idx, frame_indices = item_frame_indice[0], item_frame_indice[1:]
        
        # 获取旋转数据，可能是qpos格式或四元数格式
        rotations_data = self.rotations_list[motion_idx]
        root_quat = None  # 初始化为None，在qpos格式分支中设置
        
        if isinstance(rotations_data, dict):
            # qpos格式：直接使用qpos，不转换为旋转
            qpos = rotations_data['qpos'][frame_indices].copy().astype(self.dtype)  # (frames, 29)
            
            # 获取root_quat（如果存在）
            root_quat_raw = rotations_data.get('root_quat', None)
            if root_quat_raw is not None:
                root_quat = root_quat_raw[frame_indices].copy().astype(self.dtype)  # (frames, 4) - wxyz格式
            else:
                # 如果没有root_quat，使用单位四元数
                root_quat = np.zeros((len(frame_indices), 4), dtype=self.dtype)
                root_quat[:, 0] = 1.0  # w=1
            
            rotations = qpos[..., None]  # (frames, 29， 1)
        else:
            # 四元数格式（向后兼容）- 保持原有处理逻辑
            rotations = rotations_data[frame_indices].copy()
            # 转换为所需格式
            traj_rotation = self.local_conds['traj_pose'][motion_idx][random.choice(self.traj_aug_indexs1), frame_indices].copy()
            
            root_pos = self.root_pos_list[motion_idx][frame_indices].copy()
            root_pos[:, [0, 2]] -= root_pos[self.reference_frame_idx-1:self.reference_frame_idx, [0, 2]]
            
            traj_pos = root_pos[:, [0, 2]].copy()
            random_option = np.random.random()
            if random_option <0.5:
                pass
            elif random_option < 0.75:
                traj_pos = gaussian_filter1d(traj_pos, 5, axis=0)
            else:
                traj_pos = gaussian_filter1d(traj_pos, 10, axis=0)
            traj_pos -= traj_pos[self.reference_frame_idx-1:self.reference_frame_idx]
            
            traj_rotation = traj_rotation[self.reference_frame_idx:]
            traj_pos = traj_pos[self.reference_frame_idx:]
            
            rotation_xyzw, traj_rotation_xyzw = rotations[..., [1, 2, 3, 0]], traj_rotation[..., [1, 2, 3, 0]]
            theta = np.repeat(np.random.uniform(0, 2*np.pi), rotations.shape[0]).astype(self.dtype)
            rot_vec = R.from_rotvec(np.pad(theta[..., np.newaxis], ((0, 0), (1, 1)), 'constant', constant_values=0))
            rotations[:, 0] = (rot_vec*R.from_quat(rotation_xyzw[:, 0])).as_quat()[..., [3, 0, 1, 2]]
            traj_rotation = (rot_vec[self.reference_frame_idx:]*R.from_quat(traj_rotation_xyzw)).as_quat()[..., [3, 0, 1, 2]]
            root_pos = rot_vec.apply(root_pos).astype(self.dtype)
            traj_pos_3d = np.zeros((traj_pos.shape[0], 3), dtype=self.dtype)
            traj_pos_3d[:, [0, 2]] = traj_pos
            traj_pos = rot_vec[self.reference_frame_idx:].apply(traj_pos_3d)[:, [0, 2]].astype(self.dtype)
            
            rotations = nn_transforms.get_rotation(rotations.astype(self.dtype), self.rot_req)
            traj_rotation = nn_transforms.get_rotation(traj_rotation.astype(self.dtype), self.rot_req)
            
            root_pos_extra_dim = np.zeros((root_pos.shape[0], 1, self.per_rot_feat - 3), dtype=self.dtype)
            root_pos_extra_dim = torch.from_numpy(np.concatenate((root_pos[:, np.newaxis], root_pos_extra_dim), axis=2, dtype=self.dtype))
            rotations_with_root = torch.cat((rotations, root_pos_extra_dim), axis=1)
            
            future_motion = rotations_with_root[self.reference_frame_idx:]
            past_motion = rotations_with_root[:self.reference_frame_idx]
            
            # 直接使用预计算的CLIP特征
            text_feature = self.global_conds['text_feature'][motion_idx]  # (512,) numpy array
            text_feature = torch.from_numpy(text_feature).float()  # 转换为tensor
            
            # 确保traj_rotation是numpy array
            if isinstance(traj_rotation, torch.Tensor):
                traj_rotation_np = traj_rotation.cpu().numpy()
            else:
                traj_rotation_np = traj_rotation
            
            return {
                'data': future_motion,
                'conditions': {
                    'past_motion': past_motion,
                    'text_feature': text_feature,
                    'traj_pose': torch.from_numpy(traj_rotation_np),
                    'traj_trans': torch.from_numpy(traj_pos),
                    'mask': torch.ones(future_motion.shape[0], dtype=torch.bool)
                }
            }
        
        # qpos格式的处理
        traj_rotation = self.local_conds['traj_pose'][motion_idx][random.choice(self.traj_aug_indexs1), frame_indices].copy()
        
        root_pos = self.root_pos_list[motion_idx][frame_indices].copy()
        root_pos[:, [0, 2]] -= root_pos[self.reference_frame_idx-1:self.reference_frame_idx, [0, 2]]
        
        traj_pos = root_pos[:, [0, 2]].copy()
        random_option = np.random.random()
        if random_option <0.5:
            pass
        elif random_option < 0.75:
            traj_pos = gaussian_filter1d(traj_pos, 5, axis=0)
        else:
            traj_pos = gaussian_filter1d(traj_pos, 10, axis=0)
        traj_pos -= traj_pos[self.reference_frame_idx-1:self.reference_frame_idx]

        # Random rotation augmentation (仿照四元数格式的处理)
        # 需要在切片之前进行旋转增强，使用完整长度的数据
        full_length = len(root_pos)
        traj_rotation_xyzw = traj_rotation[..., [1, 2, 3, 0]]  # wxyz -> xyzw
        theta = np.repeat(np.random.uniform(0, 2*np.pi), full_length).astype(self.dtype)
        rot_vec = R.from_rotvec(np.pad(theta[..., np.newaxis], ((0, 0), (1, 1)), 'constant', constant_values=0))
        traj_rotation = (rot_vec*R.from_quat(traj_rotation_xyzw)).as_quat()[..., [3, 0, 1, 2]]  # xyzw -> wxyz
        root_pos = rot_vec.apply(root_pos).astype(self.dtype)
        
        # 对root_quat应用旋转增强（qpos格式）
        # root_quat已经在qpos格式分支中初始化了
        root_quat_xyzw = root_quat[..., [1, 2, 3, 0]]  # wxyz -> xyzw
        root_quat = (rot_vec*R.from_quat(root_quat_xyzw)).as_quat()[..., [3, 0, 1, 2]]  # xyzw -> wxyz
        root_quat = root_quat.astype(self.dtype)
        
        traj_pos_3d = np.zeros((full_length, 3), dtype=self.dtype)
        traj_pos_3d[:, [0, 2]] = traj_pos
        traj_pos = rot_vec.apply(traj_pos_3d)[:, [0, 2]].astype(self.dtype)
        
        # 注意：这里不切片，保持完整数据，最后再统一切片（与四元数格式分支保持一致）
        # 将traj_rotation转换为训练所需的格式（6d格式） - 先转换完整数据
        traj_rotation_full = traj_rotation.copy()
        traj_rotation_full = nn_transforms.get_rotation(traj_rotation_full.astype(self.dtype), self.rot_req)
        
        # qpos格式：直接使用，转换为tensor
        # rotations是(frames, 29, 1)，完整长度
        # root_pos和root_quat也都是完整长度
        rotations_tensor = torch.from_numpy(rotations.astype(self.dtype))  # (frames, 29, 1) - 完整长度
        root_pos_tensor = torch.from_numpy(root_pos.astype(self.dtype))  # (frames, 3) - 完整长度
        root_quat_tensor = torch.from_numpy(root_quat.astype(self.dtype))  # (frames, 4) - 完整长度
        
        # 合并：qpos + root_pos + root_quat = (frames, features, 1)
        # rotations是(frames, 29, 1)，root_pos reshape为(frames, 3, 1)，root_quat reshape为(frames, 4, 1)
        root_pos_reshaped = root_pos_tensor[..., np.newaxis]  # (frames, 3, 1)
        root_quat_reshaped = root_quat_tensor[..., np.newaxis]  # (frames, 4, 1)
        all_features = torch.cat([rotations_tensor, root_pos_reshaped, root_quat_reshaped], dim=1)  # (frames, 36, 1) - 完整长度
        
        # all_features现在是(frames, 36, 1)，其中29个是qpos，3个是root_pos，4个是root_quat
        rotations_with_root = all_features  # (frames, 36, 1) - 完整长度
        
        # 现在切片：past和future
        future_motion = rotations_with_root[self.reference_frame_idx:]
        past_motion = rotations_with_root[:self.reference_frame_idx]
        
        # 切片traj_rotation和traj_pos（用于conditions）
        traj_rotation = traj_rotation_full[self.reference_frame_idx:]
        traj_pos = traj_pos[self.reference_frame_idx:]
    
        # 直接使用预计算的CLIP特征
        text_feature = self.global_conds['text_feature'][motion_idx]  # (512,) numpy array
        text_feature = torch.from_numpy(text_feature).float()  # 转换为tensor
        
        # 确保traj_rotation是numpy array
        if isinstance(traj_rotation, torch.Tensor):
            traj_rotation_np = traj_rotation.cpu().numpy()
        else:
            traj_rotation_np = traj_rotation
        
        return {
            'data': future_motion,
            'conditions': {
                'past_motion': past_motion,
                'traj_pose': torch.from_numpy(traj_rotation_np),
                'traj_trans': torch.from_numpy(traj_pos),
                'text_feature': text_feature,
                'mask': torch.ones(future_motion.shape[0], dtype=torch.bool)
            }
        }     



# Case test for the dataset class
if __name__ == '__main__':
    
    import time
    import torch
    
    pkl_path = 'data/pkls/100style.pkl'
    rot_req = '6d'
    train_data = MotionDataset(pkl_path, rot_req, offset_frame=1, past_frame=10, future_frame=45, dtype=np.float32)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1024, shuffle=True, num_workers=8, drop_last=False, pin_memory=True)
    
    do_export_test = False
    do_loop_test = True
    
    if do_export_test:
        T_pose = train_data.T_pose
        data = train_dataloader.__iter__().__next__()
        rotations = nn_transforms.repr6d2quat(data['rotations']).numpy()
        root_pos = data['root_pos'].numpy()
         
        for i in range(10):
            T_pose_template = T_pose.copy()
            T_pose_template.rotations = rotations[i]
            T_pose_template.positions = np.zeros((rotations[i].shape[0], T_pose_template.positions.shape[1], T_pose_template.positions.shape[2]))
            T_pose_template.positions[:, 0] = root_pos[i]
            T_pose_template.export('save/visualization/example_bvh/%s.bvh' % i, save_ori_scal=True)  
             
    if do_loop_test:
        times = []
        start_time = time.time()
        for data in train_dataloader:
            end_time = time.time()
            print('Data loading time for each iteration: %ss' % (end_time - start_time))
            times.append(end_time - start_time)
            start_time = end_time
        avg_time = sum(times) / len(times)
        print('Average data loading time: %ss' % avg_time)
        print('Entire data loading time: %ss' % sum(times))
    