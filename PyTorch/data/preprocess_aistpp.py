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

from utils.g1_motion import G1Motion


def audio_feat_path_from_motion_path(motion_rtg_pkl_path: str, jukebox_dir: str) -> str:
    """
    motion: .../motions_g1/XXXX_rtg.pkl
    audio : .../jukebox_feats/XXXX.npy
    """
    base = os.path.basename(motion_rtg_pkl_path)
    if not base.endswith('_rtg.pkl'):
        raise ValueError(f'Unexpected motion filename: {base}')
    audio_name = base.replace('_rtg.pkl', '.npy')
    return os.path.join(jukebox_dir, audio_name)


if __name__ == '__main__':
    motions_dir = '/root/workspace/CAMDM/PyTorch/data/aistpp_g1/motions_g1'
    jukebox_dir = '/root/workspace/CAMDM/PyTorch/data/aistpp_g1/jukebox_feats'
    export_path = '/root/workspace/CAMDM/PyTorch/data/pkls/aistpp_g1.pkl'
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    rtg_files = sorted(glob.glob(os.path.join(motions_dir, '*_rtg.pkl')))
    if len(rtg_files) == 0:
        raise RuntimeError(f'No *_rtg.pkl found under {motions_dir}')

    if not os.path.isdir(jukebox_dir):
        raise RuntimeError(f'jukebox_feats folder not found: {jukebox_dir}')

    # (3 root pos + 4 root rot(wxyz) + 29 dof) = 36
    dof_dim = 36

    data_list = {
        'motions': []
    }

    skipped = 0
    for pkl_path in tqdm(rtg_files):
        raw = pickle.load(open(pkl_path, 'rb'))
        fps = float(raw.get('fps', 30.0))

        root_pos = np.asarray(raw['root_pos'], dtype=np.float32)  # (T,3)
        root_rot_wxyz = np.asarray(raw['root_rot'][:, [3, 0, 1, 2]], dtype=np.float32)  # xyzw -> wxyz
        dof_pos = np.asarray(raw['dof_pos'], dtype=np.float32)  # (T,29)

        dof = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=-1)
        if dof.shape[-1] != dof_dim:
            raise RuntimeError(f'Unexpected dof dim: got {dof.shape[-1]} expected {dof_dim} for {pkl_path}')

        motion = G1Motion(dof_pos=dof, fps=fps, filepath=pkl_path)

        audio_path = audio_feat_path_from_motion_path(pkl_path, jukebox_dir)
        if not os.path.exists(audio_path):
            raise RuntimeError(f'Missing audio feat: {audio_path} (for motion {pkl_path})')
        try:
            audio_feat = np.load(audio_path).astype(np.float32)
        except Exception as e:
            skipped += 1
            print(f'[WARN] skip due to broken audio feat: {audio_path} ({e})')
            continue
        if audio_feat.shape[0] != motion.frame_num:
            raise RuntimeError(
                f'Frame mismatch: motion={motion.frame_num} audio={audio_feat.shape[0]} '
                f'for motion {pkl_path} / audio {audio_path}'
            )

        motion_data = {
            'filepath': pkl_path,
            'fps': fps,
            'dof_pos': motion.dof_pos,
            'audio_feat': audio_feat,
            'text': '',
        }
        data_list['motions'].append(motion_data)

    pickle.dump(data_list, open(export_path, 'wb'))
    print(f'Finish exporting {export_path} (motions={len(data_list["motions"])}, skipped={skipped})')