#!/usr/bin/env python3
"""
视频特征提取脚本
基于 MMAudio 的视频处理流程，提取 CLIP 和 Synchformer 特征
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

from mmaudio.eval_utils import load_video, all_model_cfg
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.model.networks import get_my_mmaudio

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def extract_video_features(
    video_path: Path,
    output_dir: Path,
    variant: str = 'large_44k_v2',
    duration: float = 10.0,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    save_clip_features: bool = True,
    save_sync_features: bool = True,
    save_frames: bool = False,
):
    """
    提取视频的 CLIP 和 Synchformer 特征
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        variant: 模型变体名称
        duration: 提取的视频时长（秒）
        device: 设备 ('cuda', 'cpu', 'mps')，如果为 None 则自动选择
        dtype: 数据类型
        save_clip_features: 是否保存 CLIP 特征
        save_sync_features: 是否保存 Synchformer 特征
        save_frames: 是否保存原始帧
    """
    # 设置设备
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    log.info(f'使用设备: {device}')
    log.info(f'数据类型: {dtype}')
    
    # 加载模型配置
    if variant not in all_model_cfg:
        raise ValueError(f'未知的模型变体: {variant}，可选: {list(all_model_cfg.keys())}')
    
    model = all_model_cfg[variant]
    log.info(f'使用模型: {variant}')
    
    # 辅助函数：检查并处理路径
    def resolve_path(path: Path) -> Path:
        """解析路径，如果不存在则尝试绝对路径或下载"""
        if path.exists():
            return path
        
        log.warning(f'文件不存在: {path}')
        # 尝试使用相对于脚本目录的绝对路径
        script_dir = Path(__file__).parent
        if not path.is_absolute():
            abs_path = script_dir / path
        else:
            abs_path = path
        
        if abs_path.exists():
            log.info(f'找到绝对路径: {abs_path}')
            return abs_path
        
        # 尝试下载
        log.info(f'尝试下载: {path.name}...')
        try:
            from mmaudio.utils.download_utils import download_model_if_needed
            # 如果路径是相对的，先尝试在脚本目录下载
            if not path.is_absolute():
                download_path = script_dir / path
                download_model_if_needed(download_path)
                if download_path.exists():
                    log.info(f'下载成功: {download_path}')
                    return download_path
            # 尝试原始路径
            download_model_if_needed(path)
            if path.exists():
                log.info(f'下载成功: {path}')
                return path
        except Exception as e:
            log.warning(f'下载失败: {e}')
        
        # 如果还是不存在，返回原始路径（让后续代码报错）
        log.error(f'无法找到或下载文件: {path}')
        return path
    
    # 处理所有模型路径
    vae_path = resolve_path(model.vae_path)
    synchformer_ckpt = resolve_path(model.synchformer_ckpt)
    bigvgan_path = model.bigvgan_16k_path
    if bigvgan_path is not None:
        bigvgan_path = resolve_path(bigvgan_path)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载视频
    log.info(f'加载视频: {video_path}')
    video_info = load_video(video_path, duration)
    log.info(f'视频时长: {video_info.duration_sec:.2f} 秒')
    log.info(f'CLIP 帧数: {video_info.clip_frames.shape[0]}')
    log.info(f'Sync 帧数: {video_info.sync_frames.shape[0]}')
    
    # 初始化特征提取器
    log.info('初始化特征提取器...')
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=str(vae_path),  # 使用处理后的路径，转为字符串
        synchformer_ckpt=str(synchformer_ckpt),  # 使用处理后的路径，转为字符串
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=str(bigvgan_path) if bigvgan_path else None,  # 使用处理后的路径，转为字符串
        need_vae_encoder=False
    )
    feature_utils = feature_utils.to(device, dtype).eval()
    
    # 准备视频帧
    clip_frames = video_info.clip_frames.unsqueeze(0).to(device, dtype)  # (1, T, C, H, W)
    sync_frames = video_info.sync_frames.unsqueeze(0).to(device, dtype)  # (1, T, C, H, W)
    
    # 提取特征
    video_stem = video_path.stem
    results = {}
    
    # 提取 CLIP 特征
    if save_clip_features:
        log.info('提取 CLIP 特征...')
        with torch.inference_mode():
            clip_features = feature_utils.encode_video_with_clip(clip_frames)
        
        # 按照 MMAudio 的方式处理：detach().cpu()，然后转换为 numpy
        clip_features_np = clip_features.detach().cpu().float().numpy()
        clip_output_path = output_dir / f'{video_stem}_clip_features.npy'
        np.save(clip_output_path, clip_features_np)
        log.info(f'CLIP 特征已保存: {clip_output_path}')
        log.info(f'CLIP 特征形状: {clip_features_np.shape}')
        results['clip_features'] = {
            'path': str(clip_output_path),
            'shape': clip_features_np.shape,
            'dtype': str(clip_features_np.dtype)
        }
    
    # 提取 Synchformer 特征
    if save_sync_features:
        log.info('提取 Synchformer 特征...')
        with torch.inference_mode():
            sync_features = feature_utils.encode_video_with_sync(sync_frames)
        
        # 按照 MMAudio 的方式处理：detach().cpu()，然后转换为 numpy
        sync_features_np = sync_features.detach().cpu().float().numpy()
        sync_output_path = output_dir / f'{video_stem}_sync_features.npy'
        np.save(sync_output_path, sync_features_np)
        log.info(f'Synchformer 特征已保存: {sync_output_path}')
        log.info(f'Synchformer 特征形状: {sync_features_np.shape}')
        results['sync_features'] = {
            'path': str(sync_output_path),
            'shape': sync_features_np.shape,
            'dtype': str(sync_features_np.dtype)
        }
    
    # 保存原始帧（可选）
    if save_frames:
        log.info('保存原始帧...')
        # 按照 MMAudio 的方式处理：detach().cpu()，然后转换为 numpy
        clip_frames_np = clip_frames.squeeze(0).detach().cpu().float().numpy()
        sync_frames_np = sync_frames.squeeze(0).detach().cpu().float().numpy()
        
        clip_frames_path = output_dir / f'{video_stem}_clip_frames.npy'
        sync_frames_path = output_dir / f'{video_stem}_sync_frames.npy'
        
        np.save(clip_frames_path, clip_frames_np)
        np.save(sync_frames_path, sync_frames_np)
        
        log.info(f'CLIP 帧已保存: {clip_frames_path}')
        log.info(f'Sync 帧已保存: {sync_frames_path}')
        results['clip_frames'] = {'path': str(clip_frames_path), 'shape': clip_frames_np.shape}
        results['sync_frames'] = {'path': str(sync_frames_path), 'shape': sync_frames_np.shape}
    
    # 保存元信息
    metadata = {
        'video_path': str(video_path),
        'duration_sec': float(video_info.duration_sec),
        'clip_fps': 8.0,
        'sync_fps': 25.0,
        'clip_frame_count': int(video_info.clip_frames.shape[0]),
        'sync_frame_count': int(video_info.sync_frames.shape[0]),
        'results': results
    }
    
    import json
    metadata_path = output_dir / f'{video_stem}_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info(f'元信息已保存: {metadata_path}')
    
    return results


def batch_extract_features(
    video_dir: Path,
    output_dir: Path,
    pattern: str = '*.mp4',
    **kwargs
):
    """
    批量提取视频特征
    
    Args:
        video_dir: 视频目录
        output_dir: 输出目录
        pattern: 视频文件匹配模式
        **kwargs: 传递给 extract_video_features 的其他参数
    """
    video_dir = Path(video_dir)
    video_files = list(video_dir.glob(pattern))
    
    if not video_files:
        log.warning(f'在 {video_dir} 中未找到匹配 {pattern} 的视频文件')
        return
    
    log.info(f'找到 {len(video_files)} 个视频文件')
    
    for video_path in tqdm(video_files, desc='提取特征'):
        try:
            extract_video_features(video_path, output_dir, **kwargs)
        except Exception as e:
            log.error(f'处理 {video_path} 时出错: {e}')
            continue
    
    log.info('批量提取完成')


def main():
    parser = argparse.ArgumentParser(description='提取视频特征')
    parser.add_argument('--video', type=Path, help='视频文件路径')
    parser.add_argument('--video_dir', type=Path, help='视频目录（批量处理）')
    parser.add_argument('--output', type=Path, required=True, help='输出目录')
    parser.add_argument('--variant', type=str, default='large_44k_v2',
                        choices=list(all_model_cfg.keys()),
                        help='模型变体')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='提取的视频时长（秒）')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'],
                        help='设备（默认自动选择）')
    parser.add_argument('--full_precision', action='store_true',
                        help='使用 float32（默认使用 bfloat16）')
    parser.add_argument('--no_clip', action='store_true',
                        help='不提取 CLIP 特征')
    parser.add_argument('--no_sync', action='store_true',
                        help='不提取 Synchformer 特征')
    parser.add_argument('--save_frames', action='store_true',
                        help='保存原始帧数据')
    parser.add_argument('--pattern', type=str, default='*.mp4',
                        help='批量处理时的文件匹配模式')
    
    args = parser.parse_args()
    
    dtype = torch.float32 if args.full_precision else torch.bfloat16
    
    if args.video:
        # 单文件处理
        extract_video_features(
            video_path=args.video,
            output_dir=args.output,
            variant=args.variant,
            duration=args.duration,
            device=args.device,
            dtype=dtype,
            save_clip_features=not args.no_clip,
            save_sync_features=not args.no_sync,
            save_frames=args.save_frames
        )
    elif args.video_dir:
        # 批量处理
        batch_extract_features(
            video_dir=args.video_dir,
            output_dir=args.output,
            pattern=args.pattern,
            variant=args.variant,
            duration=args.duration,
            device=args.device,
            dtype=dtype,
            save_clip_features=not args.no_clip,
            save_sync_features=not args.no_sync,
            save_frames=args.save_frames
        )
    else:
        parser.error('必须指定 --video 或 --video_dir')


if __name__ == '__main__':
    main()