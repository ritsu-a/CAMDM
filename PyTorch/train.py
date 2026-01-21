import os
import time
import torch
import shutil
import argparse
import utils.common as common

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.logger import Logger
from network.models import MotionDiffusion
from network.training import MotionTrainingPortal
from network.dataset import AISTPPG1Dataset, G1ML3DDataset

from diffusion.create_diffusion import create_gaussian_diffusion
from config.option import add_model_args, add_train_args, add_diffusion_args, config_parse


def setup_ddp():
    """
    Enable torch.distributed if launched with torchrun.
    Returns (is_ddp, rank, world_size, local_rank).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank

def train(config, resume, logger, tb_writer):
    
    common.fixseed(1024)
    np_dtype = common.select_platform(32)
    
    print("Loading dataset..")
    # Auto-detect dataset type based on data file path or content
    # If data path contains 'g1ml3d', use G1ML3DDataset, otherwise use AISTPPG1Dataset
    if 'g1ml3d' in config.data.lower():
        print(f"Detected G1ML3D dataset, using G1ML3DDataset")
        train_data = G1ML3DDataset(
            config.data,
            offset_frame=config.arch.offset_frame,
            past_frame=config.arch.past_frame,
            future_frame=config.arch.future_frame,
            dtype=np_dtype,
            limited_num=config.trainer.load_num,
        )
    else:
        print(f"Detected AIST++ dataset, using AISTPPG1Dataset")
        train_data = AISTPPG1Dataset(
            config.data,
            offset_frame=config.arch.offset_frame,
            past_frame=config.arch.past_frame,
            future_frame=config.arch.future_frame,
            dtype=np_dtype,
            limited_num=config.trainer.load_num,
        )
    sampler = None
    if getattr(config, "ddp", False):
        sampler = DistributedSampler(train_data, num_replicas=config.world_size, rank=config.rank, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(
        train_data,
        batch_size=config.trainer.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.trainer.workers,
        drop_last=False,
        pin_memory=True,
    )
    logger.info('\nTraining Dataset includins %d clip, with %d frame per clip;' % (len(train_data), config.arch.clip_len))
    
    diffusion = create_gaussian_diffusion(config)
    
    input_feats = train_data.input_feats

    model = MotionDiffusion(
        input_feats,
        nstyles=len(getattr(train_data, 'style_set', [0])),
        njoints=train_data.joint_num,
        nfeats=train_data.per_rot_feat,
        rot_req=config.arch.rot_req,
        clip_len=config.arch.clip_len,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        device=config.device,
        audio_feat_dim=train_data.audio_dim,
    ).to(config.device)

    if getattr(config, "ddp", False):
        # DDP expects model on correct device already.
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=False)
    
    # logger.info('\nModel structure: \n%s' % str(model))
    trainer = MotionTrainingPortal(config, model, diffusion, train_dataloader, logger, tb_writer)
    
    if resume is not None:
        try:
            trainer.load_checkpoint(resume)
        except FileNotFoundError:
            print('No checkpoint found at %s' % resume); exit()
    
    trainer.run_loop()


if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='### Generative Locamotion Training')
    
    # Runtime parameters
    parser.add_argument('-n', '--name', default='debug', type=str, help='The name of this training')
    parser.add_argument('-c', '--config', default='./config/default.json', type=str, help='config file path (default: None)')
    parser.add_argument('-i', '--data', default='data/pkls/aistpp_g1.pkl', type=str)
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--save', default='./save', type=str, help='show the debug information')
    parser.add_argument('--cluster', action='store_true', help='train with GPU cluster')    
    add_model_args(parser); add_diffusion_args(parser); add_train_args(parser)
    
    args = parser.parse_args()

    ddp, rank, world_size, local_rank = setup_ddp()
    
    if args.cluster:
        # If the 'cluster' argument is provided, modify the 'data' and 'save' path to match your own cluster folder location
        args.data = 'xxxxxxx/pkls/' + args.data.split('/')[-1]
        args.save = 'xxxxx'
    
    if args.config:
        config = config_parse(args)
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    config.ddp = ddp
    config.rank = rank
    config.world_size = world_size
    config.local_rank = local_rank
    config.is_main = (rank == 0)

    if 'debug' in args.name:
        config.arch.offset_frame = config.arch.clip_len
        config.trainer.workers = 1
        config.trainer.load_num = -1
        config.trainer.batch_size = 256


    if not args.cluster:
        if config.is_main and os.path.exists(config.save) and 'debug' not in args.name and args.resume is None:
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
    
    if config.is_main:
        os.makedirs(config.save, exist_ok=True)
    if config.ddp:
        dist.barrier()

    # Only rank0 writes logs/events to disk.
    logger = Logger('%s/log.txt' % config.save) if config.is_main else Logger(os.devnull)
    tb_writer = SummaryWriter(log_dir='%s/runtime' % config.save) if config.is_main else None
    
    if torch.cuda.is_available():
        config.device = torch.device(f"cuda:{config.local_rank}" if config.ddp else "cuda")
    else:
        config.device = torch.device("cpu")

    if config.is_main:
        with open('%s/config.json' % config.save, 'w') as f:
            f.write(str(config))
        f.close() 
    
    logger.info('\Generative locamotion training with config: \n%s' % config)
    train(config, args.resume, logger, tb_writer)
    if config.is_main:
        logger.info('\nTotal training time: %s mins' % ((time.time() - start_time) / 60))

    if config.ddp:
        dist.barrier()
        dist.destroy_process_group()
    
    
