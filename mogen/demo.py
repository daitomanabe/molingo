import sys
sys.path.append('.')

import os
import argparse
import torch
import math
import copy
import clip
import smplx
import numpy as np

import torch.nn.functional as F

from os.path import join as pjoin
from argparse import Namespace
from torch.distributions.categorical import Categorical

import mogen.models.molingo.molingo as molingo_models
from mogen.models.length_estimator import LengthEstimator
from mogen.utils.fixseed import fixseed
from mogen.utils.get_opt import get_opt
from mogen.eval_mogen import load_vae_model
from mogen.utils.paramUtil import t2m_kinematic_chain
from mogen.utils.plot_script import plot_3d_motion
from mogen.utils.ms_utils import recover_from_local_rotation, smpl85_2_smpl322


def clip_encode_text(raw_text, device='cuda'):
    text = clip.tokenize(raw_text, truncate=True).to(device)
    feat_clip_text = clip_model.encode_text(text).float()
    return feat_clip_text


def load_and_freeze_clip(clip_version):
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                            jit=False)  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model


def load_len_estimator(device):
    ckpt_path = 'mogen/checkpoints/t2m/length_estimator/model/finest.tar'
    model = LengthEstimator(512, 50)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model


def pose_272_to_smpl(data_272):
    smpl_85_data = recover_from_local_rotation(data_272, 22)  # get the 85-dim smpl data
    if len(smpl_85_data.shape) == 3:
        smpl_85_data = np.squeeze(smpl_85_data, axis=0)

    pose = smpl85_2_smpl322(smpl_85_data)

    assert pose.shape[1] == 322
    use_flame = (pose.shape[1] == 322)

    rots = pose[:, :66].reshape(-1, 22, 3)

    if use_flame:
        trans = pose[:, 309:309 + 3]
    else:
        trans = pose[:, 159:159 + 3]

    trans = trans.reshape(-1, 3)
    return trans, rots


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configurations for motion generation")
    parser.add_argument("-i", "--input", type=str, default="assets/example.txt")
    parser.add_argument("-b", "--bm_path", type=str, default='/home/hynann/data/smplx_models', help="Modify with your own SMPL model directory")
    parser.add_argument("-dr", "--data_root", type=str, default='/home/hynann/data', help="Modify with your own dataset directory")
    parser.add_argument("-s", "--step", type=int, default=32, help="The number of Rectified Flow sampling steps")
    parser.add_argument("-c", "--cfg", type=float, default=4.0, help="CFG value")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="CFG temprature")
    parser.add_argument("-a", "--acc", type=int, default=3,
                        help="Sampling reduction factor. Number of steps = total_tokens // acc (e.g., 49 tokens with acc=3 → 16 steps).")
    parser.add_argument("-r", "--repeat", type=int, default=5)
    parser.add_argument('-st', "--store_smpl", action='store_true')


    args = parser.parse_args()

    fixseed(3407)

    device = 'cuda'
    text_path = args.input
    data_root = args.data_root
    step = args.step
    cfg = args.cfg
    acc = args.acc
    repeat_times = args.repeat

    save_dir = pjoin('animation', f'dim_272_cfg_{cfg}_acc_{acc}_step_{step}')
    os.makedirs(save_dir, exist_ok=True)


    opt = Namespace()
    opt.dataset_name = 'ms'

    # create eval output file
    model_dir = pjoin('mogen/checkpoints', opt.dataset_name, f'pretrained_model_272')
    opt_path = pjoin(model_dir, 'opt.txt')
    model_opt = get_opt(opt_path, device)

    joints_num = 22

    # load vae
    vae_opt_path = pjoin('mogen/checkpoints', opt.dataset_name, model_opt.vae, 'opt.txt')
    vae_ckpt_path = pjoin('mogen/checkpoints', opt.dataset_name, model_opt.vae, 'model', 'net_best_fid.ckpt')
    vae_opt = get_opt(vae_opt_path, device=device)
    vae_model = load_vae_model(vae_opt, vae_ckpt_path, 272, device=device)

    vae_embed_dim = vae_opt.output_emb_width
    ds_rate = math.pow(2, vae_opt.down_t)
    ds_rate = int(ds_rate)

    data_root = pjoin(data_root, 'HumanML3D_272')
    mean = np.load(pjoin(data_root, 'mean_std', 'Mean.npy'))
    std = np.load(pjoin(data_root, 'mean_std', 'Std.npy'))
    max_motion_length = 300
    fps = 30

    def inv_transform(data):
        return data * std + mean

    # molingo model initialization
    model_func_name = f'molingo_{model_opt.model_size}'
    molingo_func = getattr(molingo_models, model_func_name)
    partial_molingo = molingo_func()

    molingo_model = partial_molingo(vae_embed_dim=vae_embed_dim,
                              token_size=max_motion_length // ds_rate,
                              unit_length=ds_rate,
                              sample_steps=step,
                              t5_max_len=model_opt.t5_max_len,
                              adapter_layers=model_opt.aligner_layers,
                              ae=vae_opt.ae,
                              )
    molingo_model.to(device)
    model_without_ddp = molingo_model

    # load molingo model
    checkpoint = torch.load(pjoin(model_dir, f"net_best_fid.pth"), map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    model_params = list(model_without_ddp.parameters())
    ema_state_dict = checkpoint['model_ema']
    ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
    del checkpoint
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = ema_params[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)
    del ema_state_dict
    model_without_ddp.eval()

    # load clip and length estimator
    clip_version = 'ViT-B/32'
    clip_model = load_and_freeze_clip(clip_version).cuda()

    length_estimator = load_len_estimator(device)
    length_estimator.to(device)

    # load prompt list
    prompt_list = []
    length_list = []

    est_length = False
    with open(text_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            infos = line.split('#')
            prompt_list.append(infos[0])
            if len(infos) == 1 or (not is_number(infos[1])):
                est_length = True
                length_list = []
            else:
                length_list.append(float(infos[-1])) #

    # calculate text embeding
    if est_length:
        print("Since no motion length are specified, we will use estimated motion lengthes!!")
        text_embedding = clip_encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
        token_lens = Categorical(probs).sample()  # (b, seqlen)
    else:
        token_lens = torch.LongTensor(length_list) * 20 // 4 # sorry for this as the length estimator was trained under 20 FPS setup
        token_lens = token_lens.to(device).long()

    m_length = token_lens * 4 * 1.5 # 20 FPS -> 30 FPS
    m_length = m_length.int()
    captions = prompt_list

    sample = 0
    kinematic_chain = t2m_kinematic_chain

    for r in range(repeat_times):
        print("-->Repeat %d" % r)
        with torch.no_grad():
            sampled_tokens = model_without_ddp.sample_tokens(bsz=len(prompt_list), cfg=cfg, m_lens=m_length,
                                                                 cfg_schedule='constant', labels=captions,
                                                                 temperature=1.0, acc_ratio=acc)
            feats_rst = vae_model.decode(sampled_tokens / model_opt.std_factor).detach().cpu().numpy()
            data = inv_transform(feats_rst)

            for k, (caption, curr_data) in enumerate(zip(captions, data)):
                print("---->Sample %d: %s %d" % (k, caption, m_length[k]))

                curr_data = curr_data[:m_length[k]]
                trans, rots = pose_272_to_smpl(curr_data)  # [T, 3] [T, 22, 3]

                bm = smplx.create(args.bm_path, model_type='smplh', num_betas=10, gender='neutral',
                                  batch_size=trans.shape[0]).to(device='cuda')

                bparam = {}
                bparam['transl'] = torch.from_numpy(trans).float().cuda()
                bparam['global_orient'] = torch.from_numpy(rots[:, 0]).float().cuda()
                bparam['body_pose'] = torch.from_numpy(rots[:, 1:]).float().cuda().reshape(-1, 63)
                body_pred = bm(return_verts=True, **bparam)
                joint = body_pred.joints[:, :22].detach().cpu().numpy()

                video_save_path = pjoin(save_dir, "molingo_sample%d_repeat%d.mp4" % (k, r))
                plot_3d_motion(video_save_path, kinematic_chain, joint, title=caption, fps=fps)