import os
import sys
sys.path.append('./')

import torch
import numpy as np
import argparse
import utils.common as common
from types import SimpleNamespace

from torch.utils.data import DataLoader
from utils.logger import Logger
from network.models import MotionDiffusion
from network.training import MotionTrainingPortal
from network.dataset import MotionDataset
from diffusion.create_diffusion import create_gaussian_diffusion


def test_model(checkpoint_path, data_path, config_path, output_dir='./test_output', num_samples=10, past_frame=0, future_frame=128):
    """
    测试训练好的模型
    """
    common.fixseed(1024)
    np_dtype = common.select_platform(32)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Loading dataset from {data_path}")
    
    # 解析配置（先读取配置，再使用配置参数创建数据集）
    # 从checkpoint目录读取配置
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.exists(os.path.join(checkpoint_dir, 'config.json')):
        # 读取保存的配置
        with open(os.path.join(checkpoint_dir, 'config.json'), 'r') as f:
            content = f.read()
            # 替换namespace为SimpleNamespace
            content = content.replace('namespace', 'SimpleNamespace')
            # 安全地eval配置，提供所需的模块和类
            import numpy as np
            eval_globals = {
                'SimpleNamespace': SimpleNamespace,
                'namespace': SimpleNamespace,
                'np': np,
                'torch': torch,
                'device': torch.device,
            }
            config_dict = eval(content, eval_globals)
        # 如果命令行提供了past_frame或future_frame，则覆盖配置中的值
        if 'arch' in config_dict:
            if past_frame is not None:
                config_dict['arch']['past_frame'] = past_frame
            if future_frame is not None:
                config_dict['arch']['future_frame'] = future_frame
            # 重新计算clip_len
            config_dict['arch']['clip_len'] = config_dict['arch']['past_frame'] + config_dict['arch']['future_frame']
        # 确保trainer配置包含所有必需的字段
        if 'trainer' not in config_dict:
            config_dict['trainer'] = {}
        # 补充缺失的trainer字段
        trainer_defaults = {
            'lr': 0.0001,
            'lr_anneal_steps': 0,
            'epoch': 100,
            'save_freq': 5,
            'weight_decay': 0.0,
            'ema': True
        }
        for key, default_value in trainer_defaults.items():
            if key not in config_dict['trainer']:
                config_dict['trainer'][key] = default_value
    else:
        # 使用默认配置（匹配训练配置：camdm_humanml3d）
        clip_len = past_frame + future_frame
        config_dict = {
            'arch': {
                'rot_req': '6d',
                'decoder': 'trans_enc',
                'latent_dim': 256,
                'ff_size': 1024,
                'num_heads': 4,
                'num_layers': 4,
                'offset_frame': 1,
                'past_frame': past_frame,
                'future_frame': future_frame,
                'clip_len': clip_len,
                'local_cond': 'traj',
                'global_cond': 'style'
            },
            'diff': {
                'noise_schedule': 'cosine',
                'diffusion_steps': 4,
                'sigma_small': True
            },
            'trainer': {
                'batch_size': num_samples,
                'cond_mask_prob': 0.15,
                'use_loss_3d': False,
                'use_loss_contact': False,
                'use_loss_mse': True,
                'use_loss_vel': True,
                'lr': 0.0001,
                'lr_anneal_steps': 0,
                'epoch': 100,
                'save_freq': 5,
                'weight_decay': 0.0,
                'ema': True
            }
        }
    
    # 创建配置对象
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})
        return d
    
    config = dict_to_namespace(config_dict)
    config.data = data_path
    config.save = output_dir
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用配置中的参数加载数据集
    dataset_past_frame = config.arch.past_frame
    dataset_future_frame = config.arch.future_frame
    
    # 先加载完整数据集以获取正确的样式数量（不限制数量）
    full_data = MotionDataset(data_path, config.arch.rot_req, config.arch.offset_frame, 
                              dataset_past_frame, dataset_future_frame, dtype=np_dtype, limited_num=-1)
    num_styles = len(full_data.style_set)
    print(f"Found {num_styles} styles in dataset")
    print(f"Using past_frame={dataset_past_frame}, future_frame={dataset_future_frame}, clip_len={config.arch.clip_len}, rot_req={config.arch.rot_req}")
    
    # 然后加载测试数据集（限制数量）
    test_data = MotionDataset(data_path, config.arch.rot_req, config.arch.offset_frame,
                              dataset_past_frame, dataset_future_frame, dtype=np_dtype, limited_num=10)
    test_dataloader = DataLoader(test_data, batch_size=num_samples, shuffle=False, num_workers=2, drop_last=False)
    
    # 创建模型 - 使用完整数据集的样式数量
    diffusion = create_gaussian_diffusion(config)
    

    input_feats = 36
    # 实际关节数：29个qpos关节，加上root_pos算1个关节，总共30个
    # 但特征维度是32，因为root_pos有3个维度
    actual_joint_num = 30
    actual_per_rot_feat = 1


    model = MotionDiffusion(input_feats,
                actual_joint_num+1, actual_per_rot_feat,
                config.arch.rot_req, config.arch.clip_len,
                config.arch.latent_dim, config.arch.ff_size,
                config.arch.num_layers, config.arch.num_heads,
                arch=config.arch.decoder, cond_mask_prob=config.trainer.cond_mask_prob,
                device=config.device).to(config.device)
    
    # 创建logger
    logger = Logger(os.path.join(output_dir, 'test_log.txt'))
    
    # 创建trainer
    trainer = MotionTrainingPortal(config, model, diffusion, test_dataloader, logger, None)
    
    # 加载checkpoint
    trainer.load_checkpoint(checkpoint_path)
    logger.info(f'Loaded checkpoint from {checkpoint_path}')
    
    # 评估采样
    trainer.evaluate_sampling(test_dataloader, 'evaluation_results')
    
    logger.info('Evaluation completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best.pt checkpoint')
    parser.add_argument('--data', type=str, default='data/pkls/g1ml3d_train.pkl', help='Path to test data pkl file')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional, will use checkpoint config if available)')
    parser.add_argument('--output', type=str, default='./test_output', help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--past_frame', type=int, default=0, help='Number of past frames (default: 0, matching training config)')
    parser.add_argument('--future_frame', type=int, default=128, help='Number of future frames (default: 128, matching training config)')
    
    args = parser.parse_args()
    
    test_model(args.checkpoint, args.data, args.config, args.output, args.num_samples, args.past_frame, args.future_frame)
