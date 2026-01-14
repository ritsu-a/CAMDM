import os
import time
import torch
import shutil
import argparse
import utils.common as common

from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.parallel
import torch.multiprocessing as mp

from utils.logger import Logger
from network.models import MotionDiffusion
from network.training import MotionTrainingPortal
from network.dataset import MotionDataset

from diffusion.create_diffusion import create_gaussian_diffusion
from config.option import add_model_args, add_train_args, add_diffusion_args, config_parse

def train(config, resume, logger, tb_writer, rank=0, world_size=1, distributed=False):
    """
    训练函数
    Args:
        config: 配置对象
        resume: 恢复训练的checkpoint路径
        logger: 日志记录器
        tb_writer: TensorBoard写入器
        rank: 当前进程的rank（分布式训练时使用）
        world_size: 总进程数（分布式训练时使用）
        distributed: 是否使用分布式训练
    """
    # 设置随机种子（分布式训练时每个进程使用不同的种子）
    seed = 1024 + rank
    common.fixseed(seed)
    np_dtype = common.select_platform(32)
    
    # 分布式训练初始化
    if distributed:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(rank)
        config.device = torch.device(f'cuda:{rank}')
        if rank == 0:
            logger.info(f'Initialized distributed training with {world_size} GPUs')
    
    if rank == 0 or not distributed:
        print("Loading dataset..")
    
    train_data = MotionDataset(config.data, config.arch.rot_req, 
                                  config.arch.offset_frame,  config.arch.past_frame, 
                                  config.arch.future_frame, dtype=np_dtype, limited_num=config.trainer.load_num)
    
    # 分布式训练时使用DistributedSampler，否则使用普通shuffle
    if distributed:
        sampler = DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False
        batch_size = config.trainer.batch_size  # 每个GPU的batch size
    else:
        sampler = None
        shuffle = True
        batch_size = config.trainer.batch_size
    
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.trainer.workers, 
        drop_last=False, 
        pin_memory=True
    )
    
    if rank == 0 or not distributed:
        logger.info('\nTraining Dataset includins %d clip, with %d frame per clip;' % (len(train_data), config.arch.clip_len))
        if distributed:
            logger.info(f'Distributed training: {world_size} GPUs, batch_size per GPU: {batch_size}, total batch_size: {batch_size * world_size}')
    
    diffusion = create_gaussian_diffusion(config)
    
    # 检查是否是qpos格式
    first_rot_data = train_data.rotations_list[0]
    is_qpos_format = isinstance(first_rot_data, dict)
    
    if is_qpos_format:
        # qpos格式：36个特征（29个qpos + 3个root_pos + 4个root_quat），每个1维
        # input_feats = 36 * 1 = 36
        input_feats = 36
        # 实际关节数：29个qpos关节，加上root_pos算1个关节，总共30个
        # 但特征维度是32，因为root_pos有3个维度
        actual_joint_num = 30
        actual_per_rot_feat = 1
    else:
        # 传统格式
        input_feats = (train_data.joint_num+1) * train_data.per_rot_feat   # use the root translation as an extra joint
        actual_joint_num = train_data.joint_num
        actual_per_rot_feat = train_data.per_rot_feat

    model = MotionDiffusion(input_feats, len(train_data.style_set),
                actual_joint_num+1, actual_per_rot_feat, 
                config.arch.rot_req, config.arch.clip_len,
                config.arch.latent_dim, config.arch.ff_size, 
                config.arch.num_layers, config.arch.num_heads, 
                arch=config.arch.decoder, cond_mask_prob=config.trainer.cond_mask_prob, device=config.device).to(config.device)
    
    # 分布式训练时使用DistributedDataParallel包装模型
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )
        if rank == 0:
            logger.info('Model wrapped with DistributedDataParallel')
    
    # logger.info('\nModel structure: \n%s' % str(model))
    trainer = MotionTrainingPortal(config, model, diffusion, train_dataloader, logger, tb_writer)
    trainer.rank = rank
    trainer.world_size = world_size
    trainer.distributed = distributed
    
    if resume is not None:
        try:
            trainer.load_checkpoint(resume)
            if rank == 0 or not distributed:
                logger.info(f'Resumed from checkpoint: {resume}')
        except FileNotFoundError:
            if rank == 0 or not distributed:
                print('No checkpoint found at %s' % resume)
            exit()
    
    trainer.run_loop()
    
    # 分布式训练结束时清理
    if distributed:
        dist.destroy_process_group()


def main_worker(rank, world_size, args, config):
    """分布式训练的worker函数"""
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 设置设备
    config.device = torch.device(f'cuda:{rank}')
    
    # 创建logger（只在rank 0上写入）
    if rank == 0:
        os.makedirs(config.save, exist_ok=True)
        logger = Logger('%s/log.txt' % config.save)
        tb_writer = SummaryWriter(log_dir='%s/runtime' % config.save)
        logger.info(f'\nGenerative locamotion training with config: \n{config}')
        logger.info(f'Starting distributed training on {world_size} GPUs')
    else:
        logger = None
        tb_writer = None
    
    # 开始训练
    train(config, args.resume, logger, tb_writer, rank=rank, world_size=world_size, distributed=True)
    
    if rank == 0:
        logger.info(f'\nDistributed training completed on {world_size} GPUs')


if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='### Generative Locamotion Training')
    
    # Runtime parameters
    parser.add_argument('-n', '--name', default='debug', type=str, help='The name of this training')
    parser.add_argument('-c', '--config', default='./config/default.json', type=str, help='config file path (default: None)')
    parser.add_argument('-i', '--data', default='data/pkls/100style.pkl', type=str)
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--save', default='./save', type=str, help='show the debug information')
    parser.add_argument('--cluster', action='store_true', help='train with GPU cluster')
    
    # 分布式训练参数
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--world_size', type=int, default=8, help='Number of GPUs for distributed training')
    parser.add_argument('--rank', type=int, default=-1, help='Rank of the current process (for distributed training)')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (set by torch.distributed.launch)')
    
    add_model_args(parser); add_diffusion_args(parser); add_train_args(parser)
    
    args = parser.parse_args()
    
    # 检查是否使用分布式训练
    use_distributed = args.distributed and torch.cuda.device_count() > 1
    
    # 如果使用torch.distributed.launch，local_rank会被设置
    if args.local_rank != -1:
        rank = args.local_rank
        world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
        use_distributed = True
    elif use_distributed:
        rank = args.rank if args.rank >= 0 else 0
        world_size = args.world_size
    else:
        rank = 0
        world_size = 1
    
    if args.cluster:
        # If the 'cluster' argument is provided, modify the 'data' and 'save' path to match your own cluster folder location
        args.data = 'xxxxxxx/pkls/' + args.data.split('/')[-1]
        args.save = 'xxxxx'
    
    if args.config:
        config = config_parse(args)
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if 'debug' in args.name:
        config.arch.offset_frame = config.arch.clip_len
        config.trainer.workers = 1
        config.trainer.load_num = -1
        config.trainer.batch_size = 256
        # 调试模式不使用分布式训练
        use_distributed = False

    # 文件检查（只在单卡或rank 0上执行）
    if not use_distributed or rank == 0:
        if not args.cluster:
            if os.path.exists(config.save) and 'debug' not in args.name and args.resume is None:
                allow_cover = input('Model file detected, do you want to replace it? (Y/N)')
                allow_cover = allow_cover.lower()
                if allow_cover == 'n':
                    exit()
                else:
                    shutil.rmtree(config.save, ignore_errors=True)
        else:
            if os.path.exists(config.save):
                if os.path.exists('%s/best.pt' % config.save):
                    args.resume = '%s/best.pt' % config.save
                else:
                    existing_pths = [val for val in os.listdir(config.save) if 'weights_' in val ]
                    if len(existing_pths) > 0:
                        epoches = [int(filename.split('_')[1].split('.')[0]) for filename in existing_pths]
                        args.resume = '%s/%s' % (config.save, 'weights_%s.pt' % max(epoches))
        
        os.makedirs(config.save, exist_ok=True)

    # 单卡训练模式
    if not use_distributed:
        logger = Logger('%s/log.txt' % config.save)
        tb_writer = SummaryWriter(log_dir='%s/runtime' % config.save)
        
        # 自动选择GPU设备
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f'Warning: {torch.cuda.device_count()} GPUs detected but distributed training not enabled. Using GPU 0.')
            config.device = torch.device("cuda:0")
        else:
            config.device = torch.device("cpu")
        
        with open('%s/config.json' % config.save, 'w') as f:
            f.write(str(config))
        f.close() 
        
        logger.info(f'\nGenerative locamotion training with config: \n{config}')
        logger.info(f'Training on device: {config.device} (Single GPU mode)')
        train(config, args.resume, logger, tb_writer, rank=0, world_size=1, distributed=False)
        logger.info(f'\nTotal training time: {(time.time() - start_time) / 60:.2f} mins')
    
    # 分布式训练模式
    else:
        # 使用torch.multiprocessing启动多个进程
        if args.local_rank == -1:  # 如果没有使用torch.distributed.launch
            mp.spawn(
                main_worker,
                args=(world_size, args, config),
                nprocs=world_size,
                join=True
            )
        else:  # 使用torch.distributed.launch时直接调用
            main_worker(rank, world_size, args, config)
    
    
