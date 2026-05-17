import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
import textwrap

if not OmegaConf.has_resolver("tuple"):
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    print("Please install safetensors: pip install safetensors")
    sys.exit(1)

from msiglip.lightning_models import LitTBPS
from msiglip.lightning_data import TBPSDataModule
from msiglip.data.bases import ImageDataset, TextDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
TOP_K = 3


# --- MODEL LOADING ---
def normalize_key(k):
    garbage = ["model.", "backbone.", "base_model.", "image_encoder.", "text_encoder.",
               "vision_model.", "text_model.", "encoder.", "module."]
    k_clean = k
    for g in garbage:
        k_clean = k_clean.replace(g, "")
    return k_clean


def aggressive_load(model, state_dict):
    model_state = model.state_dict()
    new_sd = {}
    ckpt_map = {normalize_key(k): k for k in state_dict.keys()}
    for model_k in model_state.keys():
        clean = normalize_key(model_k)
        if clean in ckpt_map:
            ckpt_v = state_dict[ckpt_map[clean]]
            if model_state[model_k].shape == ckpt_v.shape:
                new_sd[model_k] = ckpt_v
    model.load_state_dict(new_sd, strict=False)
    return model


def load_model(config_path, ckpt_path, enable_lora):
    cfg = OmegaConf.load(config_path)
    if not enable_lora and "lora" in cfg:
        del cfg["lora"]

    dm = TBPSDataModule(cfg)
    dm.setup(stage='test')

    model = LitTBPS(cfg, num_iters_per_epoch=1)
    if enable_lora:
        model.setup_lora(cfg.lora)

    print(f"   Loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt['state_dict'].items()}
    aggressive_load(model, state_dict)

    model.to(DEVICE).eval()
    return model, dm


# --- EXTRACTION ---
def extract_all_results(model, dm):
    """Extract top-K retrieval results for ALL test queries."""
    # 1. Gallery features
    gal_feats, gal_pids, gal_imgs = [], [], []
    loader = DataLoader(
        ImageDataset(dm.dataset.test, is_train=False,
                     image_size=dm.config.aug.img.size,
                     mean=dm.config.aug.img.mean, std=dm.config.aug.img.std),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("   Extracting gallery...")
    with torch.no_grad():
        for batch in tqdm(loader):
            imgs = batch['images'].to(DEVICE)
            f = F.normalize(model.get_image_features(imgs), dim=1)
            gal_feats.append(f.cpu())
            gal_pids.append(batch['pids'])
            gal_imgs.append(batch['images'].cpu())

    gal_feats = torch.cat(gal_feats)
    gal_pids = torch.cat(gal_pids)
    gal_imgs = torch.cat(gal_imgs)

    # 2. Query features + retrieval
    results = {}
    txt_loader = DataLoader(
        TextDataset(dm.dataset.test, tokenizer=dm.tokenizer),
        batch_size=BATCH_SIZE, shuffle=False)

    print("   Extracting queries and ranking...")
    with torch.no_grad():
        for batch in txt_loader:
            pids = batch['pids']
            inputs = {}
            for k, v in batch.items():
                if k in ['input_ids', 'attention_mask', 'caption_input_ids', 'caption_attention_mask']:
                    inputs[k] = v.to(DEVICE)

            if 'caption_input_ids' in inputs:
                inputs['input_ids'] = inputs.pop('caption_input_ids')
                inputs['attention_mask'] = inputs.pop('caption_attention_mask')

            all_feats = F.normalize(model.get_text_features(inputs), dim=1).cpu()

            for idx in range(len(pids)):
                pid = pids[idx].item()
                if pid in results:
                    continue

                feat = all_feats[idx].unsqueeze(0)
                sims = torch.matmul(feat, gal_feats.t()).squeeze()
                topk_vals, topk_idx = torch.topk(sims, k=TOP_K)

                top_imgs = [gal_imgs[i] for i in topk_idx]
                top_pids = [gal_pids[i].item() for i in topk_idx]

                gt_idx = (gal_pids == pid).nonzero(as_tuple=True)[0]
                gt_img = gal_imgs[gt_idx[0]] if len(gt_idx) > 0 else None

                if 'caption_input_ids' in batch:
                    raw_ids = batch['caption_input_ids'][idx]
                else:
                    raw_ids = batch['input_ids'][idx]
                caption = dm.tokenizer.decode(raw_ids, skip_special_tokens=True)

                results[pid] = {
                    'caption': caption,
                    'top_imgs': top_imgs,
                    'top_pids': top_pids,
                    'top_sims': topk_vals.tolist(),
                    'gt_img': gt_img,
                    'is_correct': [p == pid for p in top_pids],
                    'rank1_correct': top_pids[0] == pid,
                }

    return results


# --- DETECTION ---
def detect_flipped_cases(base_results, ours_results):
    """Find cases where baseline Rank@1 is WRONG but ours is CORRECT."""
    flipped_pids = []
    for pid in base_results:
        if pid not in ours_results:
            continue
        base_wrong = not base_results[pid]['rank1_correct']
        ours_right = ours_results[pid]['rank1_correct']
        if base_wrong and ours_right:
            flipped_pids.append(pid)

    print(f"\n   Found {len(flipped_pids)} flipped cases "
          f"(baseline wrong -> ours correct at Rank@1)")
    return sorted(flipped_pids)


def detect_regression_cases(base_results, ours_results):
    """Find cases where baseline Rank@1 is CORRECT but ours is WRONG (regressions)."""
    regression_pids = []
    for pid in base_results:
        if pid not in ours_results:
            continue
        base_right = base_results[pid]['rank1_correct']
        ours_wrong = not ours_results[pid]['rank1_correct']
        if base_right and ours_wrong:
            regression_pids.append(pid)

    print(f"   Found {len(regression_pids)} regression cases "
          f"(baseline correct -> ours wrong at Rank@1)")
    return sorted(regression_pids)


# --- VISUALIZATION ---
def denormalize(tensor, mean, std, target_size=(256, 128)):
    # Dùng bilinear để resize sắc nét hơn bicubic, không dùng brightness nhân ảo
    if target_size is not None:
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=target_size,
            mode='bilinear', align_corners=False).squeeze(0)
            
    mean_t = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std_t = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    
    img = tensor * std_t + mean_t
    img = img.clamp(0, 1) # Ép chuẩn về khoảng 0-1
    return img.permute(1, 2, 0).cpu().numpy()


def save_single_comparison(pid, base_data, ours_data, output_path, img_mean, img_std):
    """Save a single side-by-side comparison image for one PID."""
    cols = 1 + TOP_K + TOP_K + 1  # text + baseline topK + ours topK + GT
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 8))
    fig.patch.set_facecolor('white')

    # Text query
    ax_txt = axes[0]
    wrapped = "\n".join(textwrap.wrap(
        f"Query PID: {pid}\n\n{base_data['caption']}", width=25))
    ax_txt.text(0.5, 0.5, wrapped, ha='center', va='center',
                fontsize=11, family='serif')
    ax_txt.axis('off')

    # Baseline top-K
    for i in range(TOP_K):
        ax = axes[1 + i]
        img = denormalize(base_data['top_imgs'][i], mean=img_mean, std=img_std, target_size=(256, 128))
        ax.imshow(img) # Bỏ interpolation='bicubic' để matplotlib tự tối ưu độ nét

        is_correct = base_data['is_correct'][i]
        color = 'green' if is_correct else 'red'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4 if is_correct else 2)
        ax.set_xticks([]); ax.set_yticks([])

        ax.text(5, 25, f"Rank-{i+1}", color='white', fontweight='bold', fontsize=11,
                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor='none'))
        if i == TOP_K // 2:
            ax.set_title("Baseline (mSigLIP)", fontsize=13,
                         fontweight='bold', pad=10, family='serif')

    # Ours top-K
    for i in range(TOP_K):
        ax = axes[1 + TOP_K + i]
        img = denormalize(ours_data['top_imgs'][i], mean=img_mean, std=img_std, target_size=(256, 128))
        ax.imshow(img)

        is_correct = ours_data['is_correct'][i]
        color = 'green' if is_correct else 'red'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4 if is_correct else 2)
        ax.set_xticks([]); ax.set_yticks([])

        ax.text(5, 25, f"Rank-{i+1}", color='white', fontweight='bold', fontsize=11,
                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor='none'))
        if i == TOP_K // 2:
            ax.set_title("Ours (mSigLIP-CLoRA)", fontsize=13,
                         fontweight='bold', pad=10, family='serif')

    # Ground truth
    ax_gt = axes[-1]
    if ours_data['gt_img'] is not None:
        img_gt = denormalize(ours_data['gt_img'], mean=img_mean, std=img_std, target_size=(256, 128))
        ax_gt.imshow(img_gt)
        for spine in ax_gt.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(3)
    else:
        ax_gt.text(0.5, 0.5, "GT Missing", ha='center', va='center')
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    ax_gt.set_title("Ground Truth", fontsize=13, fontweight='bold',
                     pad=10, family='serif')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_all_comparisons(base_results, ours_results, pids, output_dir, img_mean, img_std):
    """Save individual comparison images for a list of PIDs."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n   Saving {len(pids)} comparisons to {output_dir}/")
    for pid in tqdm(pids):
        if pid not in base_results or pid not in ours_results:
            continue
        path = os.path.join(output_dir, f"pid_{pid}.png")
        save_single_comparison(pid, base_results[pid], ours_results[pid], path, img_mean, img_std)


def print_summary(base_results, ours_results):
    """Print Rank@1 accuracy summary for both models."""
    common_pids = set(base_results.keys()) & set(ours_results.keys())
    base_correct = sum(1 for p in common_pids if base_results[p]['rank1_correct'])
    ours_correct = sum(1 for p in common_pids if ours_results[p]['rank1_correct'])
    total = len(common_pids)
    print(f"\n   === Summary ({total} queries) ===")
    print(f"   Baseline Rank@1: {base_correct}/{total} ({100*base_correct/total:.2f}%)")
    print(f"   Ours     Rank@1: {ours_correct}/{total} ({100*ours_correct/total:.2f}%)")


if __name__ == "__main__":
    CONFIG_PATH = "/mnt/data/user_data/lampt/PS/code/outputs/2026-04-09/07-46-10/.hydra/config.yaml"
    CKPT_BASELINE = "/mnt/data/user_data/lampt/PS/code/epoch=56-val_score=49.15.ckpt"
    CKPT_OURS = "/mnt/data/user_data/lampt/PS/code/outputs/2026-04-09/07-46-10/checkpoints/epoch=56-val_score=52.28.ckpt"
    OUTPUT_DIR = "qualitative_results"

    # 1. Extract
    print(">>> Processing Baseline...")
    m_base, dm = load_model(CONFIG_PATH, CKPT_BASELINE, enable_lora=False)
    res_base = extract_all_results(m_base, dm)
    del m_base; torch.cuda.empty_cache()

    print("\n>>> Processing Ours...")
    m_ours, _ = load_model(CONFIG_PATH, CKPT_OURS, enable_lora=True)
    res_ours = extract_all_results(m_ours, dm)
    del m_ours; torch.cuda.empty_cache()

    # 2. Detect
    print_summary(res_base, res_ours)
    flipped = detect_flipped_cases(res_base, res_ours)
    regressions = detect_regression_cases(res_base, res_ours)

    # Lấy thông số chuẩn hóa chính xác từ config thay vì hardcode
    img_mean = dm.config.aug.img.mean
    img_std = dm.config.aug.img.std

    # 3. Save
    save_all_comparisons(res_base, res_ours, flipped,
                         os.path.join(OUTPUT_DIR, "flipped"), img_mean, img_std)
    if regressions:
        save_all_comparisons(res_base, res_ours, regressions,
                             os.path.join(OUTPUT_DIR, "regressions"), img_mean, img_std)

    print(f"\n   Done! Flipped: {len(flipped)}, Regressions: {len(regressions)}")