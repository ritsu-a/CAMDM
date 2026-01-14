import os
import numpy as np
import blobfile as bf
import utils.common as common
from tqdm import tqdm
import utils.nn_transforms as nn_transforms
import itertools

import torch
from torch.optim import AdamW
from torch.utils.data import Subset, DataLoader
from torch_ema import ExponentialMovingAverage

from diffusion.resample import create_named_schedule_sampler
from diffusion.gaussian_diffusion import *


class BaseTrainingPortal:
    def __init__(self, config, model, diffusion, dataloader, logger, tb_writer, prior_loader=None):
        
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.logger = logger
        self.tb_writer = tb_writer
        self.config = config
        self.batch_size = config.trainer.batch_size
        self.lr = config.trainer.lr
        self.lr_anneal_steps = config.trainer.lr_anneal_steps

        self.epoch = 0
        self.num_epochs = config.trainer.epoch
        self.save_freq = config.trainer.save_freq
        self.best_loss = 1e10
        
        print('Train with %d epoches, %d batches by %d batch_size' % (self.num_epochs, len(self.dataloader), self.batch_size))

        self.save_dir = config.save
        
        # 分布式训练相关属性（在train函数中设置）
        self.rank = getattr(self, 'rank', 0)
        self.world_size = getattr(self, 'world_size', 1)
        self.distributed = getattr(self, 'distributed', False)
        
        # 获取实际模型（如果是DDP包装的，返回module；否则返回模型本身）
        self.get_model = lambda: self.model.module if self.distributed else self.model

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=config.trainer.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs, eta_min=self.lr * 0.1)
        
        if config.trainer.ema:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        
        self.device = config.device

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.use_ddp = False
        
        self.prior_loader = prior_loader
        
        
    def diffuse(self, x_start, t, cond, noise=None, return_loss=False):
        raise NotImplementedError('diffuse function must be implemented')

    def evaluate_sampling(self, dataloader, save_folder_name):
        raise NotImplementedError('evaluate_sampling function must be implemented')
    
        
    def run_loop(self):
        sampling_num = 16
        sampling_idx = np.random.randint(0, len(self.dataloader.dataset), sampling_num)
        sampling_subset = DataLoader(Subset(self.dataloader.dataset, sampling_idx), batch_size=sampling_num)
        self.evaluate_sampling(sampling_subset, save_folder_name='init_samples')
        
        epoch_process_bar = tqdm(range(self.epoch, self.num_epochs), desc=f'Epoch {self.epoch}')
        for epoch_idx in epoch_process_bar:
            self.model.train()
            self.model.training = True
            self.epoch = epoch_idx
            epoch_losses = {}
            
            data_len = len(self.dataloader)
            
            for datas in self.dataloader:
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
                x_start = datas['data']

                self.opt.zero_grad()
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                
                _, losses = self.diffuse(x_start, t, cond, noise=None, return_loss=True)
                total_loss = (losses["loss"] * weights).mean()
                total_loss.backward()
                self.opt.step()
            
                if self.config.trainer.ema:
                    self.ema.update()
                
                for key_name in losses.keys():
                    if 'loss' in key_name:
                        if key_name not in epoch_losses.keys():
                            epoch_losses[key_name] = []
                        epoch_losses[key_name].append(losses[key_name].mean().item())
            
            if self.prior_loader is not None:
                for prior_datas in itertools.islice(self.prior_loader, data_len):
                    prior_datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in prior_datas.items()}
                    prior_cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in prior_datas['conditions'].items()}
                    prior_x_start = prior_datas['data']
                    
                    self.opt.zero_grad()
                    t, weights = self.schedule_sampler.sample(prior_x_start.shape[0], self.device)
                    
                    _, prior_losses = self.diffuse(prior_x_start, t, prior_cond, noise=None, return_loss=True)
                    total_loss = (prior_losses["loss"] * weights).mean()
                    total_loss.backward()
                    self.opt.step()
                    
                    for key_name in prior_losses.keys():
                        if 'loss' in key_name:
                            if key_name not in epoch_losses.keys():
                                epoch_losses[key_name] = []
                            epoch_losses[key_name].append(prior_losses[key_name].mean().item())
            
            loss_str = ''
            for key in epoch_losses.keys():
                loss_str += f'{key}: {np.mean(epoch_losses[key]):.6f}, '
            
            epoch_avg_loss = np.mean(epoch_losses['loss'])
            
            if self.epoch > 10 and epoch_avg_loss < self.best_loss:                
                self.save_checkpoint(filename='best')
            
            if epoch_avg_loss < self.best_loss:
                self.best_loss = epoch_avg_loss
            
            epoch_process_bar.set_description(f'Epoch {epoch_idx}/{self.config.trainer.epoch} | loss: {epoch_avg_loss:.6f} | best_loss: {self.best_loss:.6f}')
            if self.logger:
                self.logger.info(f'Epoch {epoch_idx}/{self.config.trainer.epoch} | {loss_str} | best_loss: {self.best_loss:.6f}')
                        
            if epoch_idx > 0 and epoch_idx % self.config.trainer.save_freq == 0:
                self.save_checkpoint(filename=f'weights_{epoch_idx}')
                self.evaluate_sampling(sampling_subset, save_folder_name='train_samples')
            
            for key_name in epoch_losses.keys():
                if 'loss' in key_name:
                    if self.tb_writer:
                        self.tb_writer.add_scalar(f'train/{key_name}', np.mean(epoch_losses[key_name]), epoch_idx)

            self.scheduler.step()
        
        best_path = '%s/best.pt' % (self.config.save)
        self.load_checkpoint(best_path)
        self.evaluate_sampling(sampling_subset, save_folder_name='best')


    def state_dict(self):
        model = self.get_model()
        model_state = model.state_dict()
        opt_state = self.opt.state_dict()
            
        return {
            'epoch': self.epoch,
            'state_dict': model_state,
            'opt_state_dict': opt_state,
            'config': self.config,
            'loss': self.best_loss,
        }

    def save_checkpoint(self, filename='weights'):
        # 分布式训练时只在rank 0上保存checkpoint
        if self.distributed and self.rank != 0:
            return
        save_path = '%s/%s.pt' % (self.config.save, filename)
        with bf.BlobFile(bf.join(save_path), "wb") as f:
            torch.save(self.state_dict(), f)
        if self.logger:
            self.logger.info(f'Saved checkpoint: {save_path}')


    def load_checkpoint(self, resume_checkpoint, load_hyper=True):
        if bf.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint)
            model = self.get_model()
            model.load_state_dict(checkpoint['state_dict'])
            if load_hyper:
                self.epoch = checkpoint['epoch'] + 1
                self.best_loss = checkpoint['loss']
                self.opt.load_state_dict(checkpoint['opt_state_dict'])
            if self.logger:
                self.logger.info('\nLoad checkpoint from %s, start at epoch %d, loss: %.4f' % (resume_checkpoint, self.epoch, checkpoint['loss']))
        else:
            raise FileNotFoundError(f'No checkpoint found at {resume_checkpoint}')


class MotionTrainingPortal(BaseTrainingPortal):
    def __init__(self, config, model, diffusion, dataloader, logger, tb_writer, finetune_loader=None):
        super().__init__(config, model, diffusion, dataloader, logger, tb_writer, finetune_loader)
        # 检查 T_pose 是否有 offsets 和 parents 属性（NPZMotion 没有这些）

 
        # NPZMotion 没有 offsets 和 parents，设置为 None
        self.skel_offset = None
        self.skel_parents = None
        self.has_skeleton_info = False
        # 如果尝试使用 3D loss，需要关闭它
        if self.config.trainer.use_loss_3d or self.config.trainer.use_loss_contact:
            if self.logger:
                self.logger.info('Warning: T_pose has no offsets/parents, disabling use_loss_3d and use_loss_contact')
            self.config.trainer.use_loss_3d = False
            self.config.trainer.use_loss_contact = False
        

    def diffuse(self, x_start, t, cond, noise=None, return_loss=False):
        # 处理qpos格式（3维）和传统格式（4维）
        if len(x_start.shape) == 3:
            # qpos格式: (batch, frame_num, joint_num * joint_feat)
            batch_size, frame_num, total_feat = x_start.shape
            # 假设joint_feat=1（qpos），需要reshape
            # 如果total_feat=36，是29个qpos + 3个root_pos + 4个root_quat
            joint_num = total_feat
            joint_feat = 1
            x_start = x_start.unsqueeze(-1)  # (batch, frame_num, joint_num, 1)
        else:
            batch_size, frame_num, joint_num, joint_feat = x_start.shape
        
        x_start = x_start.permute(0, 2, 3, 1)  # (batch, joint_num, joint_feat, frame_num)
        
        if noise is None:
            noise = th.randn_like(x_start)
        
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)
        
        # [bs, joint_num, joint_feat, future_frames]
        cond['past_motion'] = cond['past_motion'].permute(0, 2, 3, 1) # [bs, joint_num, joint_feat, past_frames]
        cond['traj_pose'] = cond['traj_pose'].permute(0, 2, 1) # [bs, 6, frame_num//2]
        cond['traj_trans'] = cond['traj_trans'].permute(0, 2, 1) # [bs, 2, frame_num//2]
        
        model = self.get_model()
        model_output = model.interface(x_t, self.diffusion._scale_timesteps(t), cond)
        
        if return_loss:
            loss_terms = {}
            
            if self.diffusion.model_var_type in [ModelVarType.LEARNED,  ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                loss_terms["vb"] = self.diffusion._vb_terms_bpd(model=lambda *args, r=frozen_out: r, x_start=x_start, x_t=x_t, t=t, clip_denoised=False)["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    loss_terms["vb"] *= self.diffusion.num_timesteps / 1000.0
            target = {
                ModelMeanType.PREVIOUS_X: self.diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.diffusion.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            mask = cond['mask'].view(batch_size, 1, 1, -1)
            
            if self.config.trainer.use_loss_mse:
                loss_terms['loss_data'] = self.diffusion.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
                
            if self.config.trainer.use_loss_vel:
                model_output_vel = model_output[..., 1:] - model_output[..., :-1]
                target_vel = target[..., 1:] - target[..., :-1]
                loss_terms['loss_data_vel'] = self.diffusion.masked_l2(target_vel[:, :-1], model_output_vel[:, :-1], mask[..., 1:])
                  
            # 只有在有 skeleton 信息（offsets 和 parents）时才计算 3D loss
            if self.has_skeleton_info and (self.config.trainer.use_loss_3d or self.config.trainer.use_loss_contact):
                target_rot, pred_rot, past_rot = target.permute(0, 3, 1, 2), model_output.permute(0, 3, 1, 2), cond['past_motion'].permute(0, 3, 1, 2)
                target_root_pos, pred_root_pos, past_root_pos = target_rot[:, :, -1, :3], pred_rot[:, :, -1, :3], past_rot[:, :, -1, :3]
                skeletons = self.skel_offset.unsqueeze(0).expand(batch_size, -1, -1)
                parents = self.skel_parents[None]
                
                target_xyz = neural_FK(target_rot[:, :, :-1], skeletons, target_root_pos, parents, rotation_type=self.config.arch.rot_req)
                pred_xyz = neural_FK(pred_rot[:, :, :-1], skeletons, pred_root_pos, parents, rotation_type=self.config.arch.rot_req)
                
                if self.config.trainer.use_loss_3d:
                    loss_terms["loss_geo_xyz"] = self.diffusion.masked_l2(target_xyz.permute(0, 2, 3, 1), pred_xyz.permute(0, 2, 3, 1), mask)
                
                if self.config.trainer.use_loss_vel and self.config.trainer.use_loss_3d:
                    target_xyz_vel = target_xyz[:, 1:] - target_xyz[:, :-1]
                    pred_xyz_vel = pred_xyz[:, 1:] - pred_xyz[:, :-1]
                    loss_terms["loss_geo_xyz_vel"] = self.diffusion.masked_l2(target_xyz_vel.permute(0, 2, 3, 1), pred_xyz_vel.permute(0, 2, 3, 1), mask[..., 1:])
                
                if self.config.trainer.use_loss_contact:
                    l_foot_idx, r_foot_idx = 24, 19
                    relevant_joints = [l_foot_idx, r_foot_idx]
                    target_xyz_reshape = target_xyz.permute(0, 2, 3, 1)  
                    pred_xyz_reshape = pred_xyz.permute(0, 2, 3, 1)
                    gt_joint_xyz = target_xyz_reshape[:, relevant_joints, :, :]  # [BatchSize, 2, 3, Frames]
                    gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
                    fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=2).repeat(1, 1, 3, 1)
                    pred_joint_xyz = pred_xyz_reshape[:, relevant_joints, :, :]  # [BatchSize, 2, 3, Frames]
                    pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
                    pred_vel[~fc_mask] = 0
                    loss_terms["loss_foot_contact"] = self.diffusion.masked_l2(pred_vel,
                                                torch.zeros(pred_vel.shape, device=pred_vel.device),
                                                mask[:, :, :, 1:])
            
            loss_terms["loss"] = loss_terms.get('vb', 0.) + \
                            loss_terms.get('loss_data', 0.) + \
                            loss_terms.get('loss_data_vel', 0.) + \
                            loss_terms.get('loss_geo_xyz', 0) + \
                            loss_terms.get('loss_geo_xyz_vel', 0) + \
                            loss_terms.get('loss_foot_contact', 0)
            
            return model_output.permute(0, 3, 1, 2), loss_terms
        
        return model_output.permute(0, 3, 1, 2)
        
    
    def evaluate_sampling(self, dataloader, save_folder_name):
        # 分布式训练时只在rank 0上评估和保存样本
        if self.distributed and self.rank != 0:
            return
        self.model.eval()
        self.model.training = False
        common.mkdir('%s/%s' % (self.save_dir, save_folder_name))
        
        datas = next(iter(dataloader)) 
        datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
        cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
        x_start = datas['data']
        t, _ = self.schedule_sampler.sample(dataloader.batch_size, self.device)
        with torch.no_grad():
            model_output = self.diffuse(x_start, t, cond, noise=None, return_loss=False)
        
        common_past_motion = cond['past_motion'].permute(0, 3, 1, 2)
        self.export_samples(x_start, common_past_motion, '%s/%s/' % (self.save_dir, save_folder_name), 'gt')
        self.export_samples(model_output, common_past_motion, '%s/%s/' % (self.save_dir, save_folder_name), 'pred')
        
        if self.logger:
            self.logger.info(f'Evaluate the sampling {save_folder_name} at epoch {self.epoch}')
        

    def export_samples(self, future_motion_feature, past_motion_feature, save_path, prefix):
        motion_feature = torch.cat((past_motion_feature, future_motion_feature), dim=1)
        
        # 检查数据格式：qpos 格式 vs 传统旋转格式
        # qpos 格式：(batch, frames, 36, 1) - 29 qpos + 3 root_pos + 4 root_quat
        # 传统格式：(batch, frames, joint_num, 6) - 6D 旋转
        batch_size, total_frames, feature_dim, feat_size = motion_feature.shape
        
        if feat_size == 1 and feature_dim == 36:
            # qpos 格式（包含root_quat）
            qpos_angles = motion_feature[:, :, :29, 0].cpu().numpy()  # (batch, frames, 29)
            root_pos = motion_feature[:, :, 29:32, 0].cpu().numpy()  # (batch, frames, 3)
            root_quat = motion_feature[:, :, 32:36, 0].cpu().numpy()  # (batch, frames, 4) - wxyz格式
            is_qpos_format = True
        else:
            # 传统 6D 旋转格式
            rotations_6d = motion_feature[:, :, :-1, :]  # (batch, frames, joint_num, 6)
            root_pos = motion_feature[:, :, -1, :3].cpu().numpy()  # (batch, frames, 3)
            
            # 将 6D 表示转换为四元数
            rotations_6d_flat = rotations_6d.reshape(-1, 6).cpu()
            rotations_quat = nn_transforms.repr6d2quat(rotations_6d_flat).numpy()
            rotations_quat = rotations_quat.reshape(batch_size, total_frames, -1, 4)  # (batch, frames, joint_num, 4)
            is_qpos_format = False
        
        if not self.has_skeleton_info:
            # NPZMotion：导出为 NPZ 格式，格式与load_npz_motion读取时一致
            # 使用默认的g1关节名称（29个关节）
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
            
            for sample_idx in range(future_motion_feature.shape[0]):
                if is_qpos_format:
                    # 直接使用 qpos 数据，不需要从四元数转换
                    joint_angles = qpos_angles[sample_idx]  # (frames, 29)
                    
                    # root_quat从feature中提取（wxyz格式）
                    root_quat_sample = root_quat[sample_idx]  # (frames, 4) - wxyz格式
                else:
                    # 从四元数转换为角度
                    root_quat_sample = rotations_quat[sample_idx, :, 0, :]  # (frames, 4) - 根关节四元数 (wxyz格式)
                    joint_quats = rotations_quat[sample_idx, :, 1:, :]  # (frames, 29, 4) - 其他关节四元数
                    
                    # 将其他关节的四元数转换为角度（绕Z轴旋转）
                    from scipy.spatial.transform import Rotation as R
                    joint_angles = np.zeros((total_frames, 29), dtype=np.float32)
                    for j in range(29):
                        quat_xyzw = joint_quats[:, j, [1, 2, 3, 0]]  # wxyz -> xyzw (scipy使用xyzw格式)
                        try:
                            # 从四元数提取Z轴旋转角度
                            r = R.from_quat(quat_xyzw)
                            angles = r.as_euler('xyz', degrees=False)  # 获取xyz欧拉角
                            joint_angles[:, j] = angles[:, 2]  # 只取Z轴角度
                        except:
                            # 如果转换失败，使用0角度
                            joint_angles[:, j] = 0.0
                
                # 注意：root_pos和root_quat是Y-up格式（因为pkl中保存的是Y-up）
                # 需要转换为Z-up格式
                # 定义坐标系转换函数（避免导入make_pose_data时触发style_helper依赖）
                from scipy.spatial.transform import Rotation as R
                
                def convert_y_up_to_z_up(positions, quaternions=None):
                    """
                    将坐标系从+Y朝上（BVH/Y-up）转换回+Z朝上（OpenGL/Z-up）
                    转换规则：(x, y, z) -> (x, -z, y)
                    对于四元数：需要相应的旋转变换（绕X轴旋转+90度）
                    """
                    # 位置转换：(x, y, z) -> (x, -z, y)
                    converted_positions = np.zeros_like(positions)
                    converted_positions[..., 0] = positions[..., 0]  # X保持不变
                    converted_positions[..., 1] = -positions[..., 2]  # Y <- -Z（前）
                    converted_positions[..., 2] = positions[..., 1]  # Z <- Y（上）
                    
                    if quaternions is not None:
                        # 确保quaternions是2D数组 (frames, 4)
                        quaternions = np.asarray(quaternions)
                        if quaternions.ndim == 1:
                            quaternions = quaternions.reshape(1, -1)
                        
                        # 四元数转换：绕X轴旋转+90度，将Y-up转换回Z-up
                        convert_quat = R.from_euler('x', np.pi/2).as_quat()  # xyzw格式
                        
                        converted_quaternions = np.zeros_like(quaternions)
                        for i in range(quaternions.shape[0]):
                            # 当前四元数（wxyz格式）转换为xyzw格式进行运算
                            q_wxyz = quaternions[i]  # (4,)
                            q_xyzw = q_wxyz[[1, 2, 3, 0]]  # wxyz -> xyzw
                            # 应用坐标转换旋转
                            q_rotated = (R.from_quat(convert_quat) * R.from_quat(q_xyzw)).as_quat()
                            # 转回wxyz格式
                            converted_quaternions[i] = q_rotated[[3, 0, 1, 2]]
                        
                        # 如果原始quaternions是1D，返回1D结果
                        if converted_quaternions.shape[0] == 1 and quaternions.ndim == 1:
                            converted_quaternions = converted_quaternions[0]
                        
                        return converted_positions, converted_quaternions
                    
                    return converted_positions
                
                # 转换根位置和根四元数（Y-up -> Z-up）
                root_pos_yup = root_pos[sample_idx]  # (frames, 3) - Y-up格式
                root_pos_yup_reshaped = root_pos_yup.reshape(-1, 1, 3)  # (frames, 1, 3)
                root_quat_yup = root_quat_sample  # (frames, 4) - Y-up格式，已经在上面索引过了
                
                # 转换（返回converted_positions, converted_quaternions）
                root_pos_zup_reshaped, root_quat_zup = convert_y_up_to_z_up(
                    root_pos_yup_reshaped,
                    root_quat_yup
                )
                root_pos_zup = root_pos_zup_reshaped.reshape(-1, 3)  # (frames, 3) - Z-up格式
                
                # 构建 qpos：根位置(3) + 根旋转四元数(4) + 关节角度(29) = 36
                # 注意：保存为Z-up格式
                qpos = np.zeros((total_frames, 36), dtype=np.float32)
                qpos[:, 0:3] = root_pos_zup  # 根位置（Z-up格式）
                qpos[:, 3:7] = root_quat_zup  # 根旋转四元数 (wxyz格式，Z-up)
                qpos[:, 7:] = joint_angles  # 29个关节的角度（不受坐标系影响）
                
                # 创建 global_body_pos: (frames, 30, 3) - 第一个是root，后面是29个关节
                # 注意：这里我们没有完整的全局位置信息，所以只保存根位置，其他设为0
                # 数据已经是Z-up格式
                global_body_pos = np.zeros((total_frames, 30, 3), dtype=np.float32)
                global_body_pos[:, 0, :] = root_pos_zup  # 根位置（Z-up格式）
                # 其他关节位置暂时设为0（如果需要可以通过FK计算）
                
                # 创建 NPZ 数据，格式与load_npz_motion读取时一致（Z-up格式）
                npz_data = {
                    'qpos': qpos,  # (frames, 36) - Z-up格式
                    'global_body_pos': global_body_pos,  # (frames, 30, 3) - Z-up格式
                    'joint_names': np.array(['root'] + joint_names, dtype=object),  # (30,)
                    'jnt_type': np.array([0] + [3] * 29, dtype=np.int32),  # (30,) - 0=free joint, 3=hinge joint
                    'frequency': 60.0,  # 60fps
                    'njnt': 30,
                }
                
                output_path = f'{save_path}/motion_{sample_idx}.{prefix}.npz'
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.savez(output_path, **npz_data)
                if self.logger:
                    self.logger.info(f'Exported NPZ (Z-up): {output_path}')
                else:
                    print(f'Exported NPZ (Z-up): {output_path}')
        else:
            # Motion类：导出为 BVH 格式
            if not is_qpos_format:
                rotations = rotations_quat  # 已经在前面转换为四元数
            else:
                # 不应该发生，qpos格式应该与has_skeleton_info=False对应
                if self.logger:
                    self.logger.warning('Unexpected: qpos format with skeleton info available')
                return
            
            for sample_idx in range(future_motion_feature.shape[0]):
                T_pose_template = self.dataloader.dataset.T_pose.copy()
                T_pose_template.rotations = rotations[sample_idx]
                T_pose_template.positions = np.zeros((rotations[sample_idx].shape[0], T_pose_template.positions.shape[1], T_pose_template.positions.shape[2]))
                T_pose_template.positions[:, 0] = root_pos[sample_idx]
                if hasattr(T_pose_template, 'export'):
                    T_pose_template.export(f'{save_path}/motion_{sample_idx}.{prefix}.bvh', save_ori_scal=True)  
        