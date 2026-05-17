import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    print("Please install safetensors: pip install safetensors")
    sys.exit(1)

from msiglip.lightning_models import LitTBPS
from msiglip.lightning_data import TBPSDataModule

# --- CẤU HÌNH ---
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
TOP_K = 100 # Lưu lại 100 ca khó nhất để dành vẽ

def setup_env():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    try:
        OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
        OmegaConf.register_new_resolver("eval", eval)
    except Exception:
        pass

def load_pretrained_backbone(config_path, pretrained_path):
    print(f"🦕 Loading Pre-trained Backbone from: {pretrained_path}")
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    
    # Init DataModule để lấy thông tin vocab/num_classes
    dm = TBPSDataModule(cfg)
    dm.setup()
    
    model = LitTBPS(cfg, num_iters_per_epoch=1)
    
    # Load Safetensors & Map Keys (Quan trọng cho mSigLIP)
    sd = load_safetensors(pretrained_path)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("vision_model"):
            new_key = k.replace("vision_model", "model.image_encoder.model.vision_model")
        elif k.startswith("text_model"):
            new_key = k.replace("text_model", "model.text_encoder.model.text_model")
        else:
            new_key = "model.image_encoder.model." + k
        new_sd[new_key] = v

    msg = model.load_state_dict(new_sd, strict=False)
    print(f"   Load Status: {msg}")
    
    model.to(DEVICE)
    model.eval()
    return model, dm

def extract_all_features(model, dataloader, desc="Extracting"):
    """Trích xuất feature theo batch để tránh tràn RAM"""
    feats = []
    pids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            # Xử lý tùy loại batch (Image hay Text)
            if 'images' in batch:
                # Image Batch
                inputs = batch['images'].to(DEVICE)
                f = model.get_image_features(inputs)
            else:
                # Text Batch
                inputs = {
                    'input_ids': batch['caption_input_ids'].to(DEVICE),
                    'attention_mask': batch['caption_attention_mask'].to(DEVICE)
                }
                f = model.get_text_features(inputs)
            
            # Normalize ngay lập tức
            f = F.normalize(f, dim=1)
            feats.append(f.cpu())
            pids.extend(batch['pids'].tolist())
            
    return torch.cat(feats), torch.tensor(pids)

def mine_hard_negatives(img_feats, img_pids, txt_feats, txt_pids):
    print("\n⛏️  Mining Hard Negatives on Full Dataset...")
    
    # Tính ma trận tương đồng lớn (Chunking nếu cần, nhưng 3k-10k ảnh thì GPU chịu được)
    # Sim = Txt @ Img.T
    sim_matrix = torch.matmul(txt_feats.to(DEVICE), img_feats.to(DEVICE).T) # (N_txt, N_img)
    
    hard_negatives_list = []
    
    img_pids_dev = img_pids.to(DEVICE)
    txt_pids_dev = txt_pids.to(DEVICE)

    # Duyệt qua từng câu Query
    for i in tqdm(range(len(txt_feats)), desc="Searching"):
        query_pid = txt_pids_dev[i]
        
        # 1. Xác định đâu là ảnh đúng (Positive), đâu là ảnh sai (Negative)
        pos_mask = (img_pids_dev == query_pid)
        neg_mask = ~pos_mask
        
        # Nếu không có ảnh Positive trong tập test (hiếm gặp), bỏ qua
        if not pos_mask.any(): continue
        
        # 2. Lấy điểm Similarity
        # Positive Sim: Lấy cái cao nhất trong số các ảnh đúng (Best Match)
        pos_sim = sim_matrix[i][pos_mask].max().item()
        
        # Negative Sims: Lấy tất cả ảnh sai
        neg_sims = sim_matrix[i][neg_mask]
        
        # 3. Tìm Hardest Negative (Ảnh sai mà có Sim cao nhất)
        hard_neg_sim, hard_neg_rel_idx = torch.max(neg_sims, dim=0)
        
        # Quy đổi lại index thực trong tập img_feats
        # Lấy indices của các ảnh negative
        neg_indices = torch.nonzero(neg_mask, as_tuple=True)[0]
        real_hard_neg_idx = neg_indices[hard_neg_rel_idx].item()
        
        # Tìm index của ảnh Positive (để vẽ tham chiếu)
        # Lấy ảnh pos có sim cao nhất
        pos_indices = torch.nonzero(pos_mask, as_tuple=True)[0]
        # (Đoạn này lấy pos đầu tiên hoặc pos best match đều được, lấy best match cho chuẩn)
        best_pos_rel_idx = torch.argmax(sim_matrix[i][pos_mask])
        real_pos_idx = pos_indices[best_pos_rel_idx].item()

        # 4. Lưu lại thông tin
        hard_negatives_list.append({
            'txt_idx': i,
            'pos_img_idx': real_pos_idx,
            'neg_img_idx': real_hard_neg_idx,
            'txt_pid': query_pid.item(),
            'neg_pid': img_pids[real_hard_neg_idx].item(),
            'pos_sim': pos_sim,
            'neg_sim': hard_neg_sim.item(),
            'diff': hard_neg_sim.item() - pos_sim # Diff càng dương lớn -> Càng sai nặng
        })
        
    # 5. Sắp xếp: Ca nào sai nặng nhất lên đầu
    # (Hard Negative Sim > Positive Sim là sai rất nặng)
    hard_negatives_list.sort(key=lambda x: x['diff'], reverse=True)
    
    return hard_negatives_list[:TOP_K]

if __name__ == "__main__":
    # --- ĐƯỜNG DẪN (SỬA LẠI NẾU CẦN) ---
    CONFIG_PATH = "/mnt/data/user_data/lampt/PS/code/outputs/2026-01-16/10-20-32/.hydra/config.yaml"
    PRETRAINED_PATH = "/mnt/data/user_data/lampt/PS/code/m_siglip_checkpoints/model.safetensors"
    SAVE_FILE = "hard_negatives_data.pt"
    
    setup_env()
    
    # 1. Load Model Backbone
    model, dm = load_pretrained_backbone(CONFIG_PATH, PRETRAINED_PATH)
    
    # 2. Extract Features (Dùng Dataloader chuẩn của DataModule)
    print("\n🚀 Extracting Features...")
    # Lưu ý: DataModule thường có test_dataloader()
    # Ta cần tách riêng Image Loader và Text Loader hoặc duyệt qua dataset
    
    # Cách an toàn nhất: Duyệt qua dataset thủ công và batching
    from torch.utils.data import DataLoader
    
    # Image Loader
    img_loader = DataLoader(dm.test_img_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    img_feats, img_pids = extract_all_features(model, img_loader, desc="Images")
    
    # Text Loader
    txt_loader = DataLoader(dm.test_txt_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    txt_feats, txt_pids = extract_all_features(model, txt_loader, desc="Texts")
    
    print(f"   Done. Images: {img_feats.shape}, Texts: {txt_feats.shape}")
    
    # 3. Mine Hard Negatives
    top_hard_cases = mine_hard_negatives(img_feats, img_pids, txt_feats, txt_pids)
    
    # 4. In thử vài kết quả
    print(f"\n😱 Top 5 Hardest Cases (Baseline):")
    for i, case in enumerate(top_hard_cases[:5]):
        print(f"   {i+1}. Query PID {case['txt_pid']} | HardNeg PID {case['neg_pid']}")
        print(f"      Pos Sim: {case['pos_sim']:.4f} vs Neg Sim: {case['neg_sim']:.4f} (Diff: {case['diff']:.4f})")
        if case['diff'] > 0:
            print("      => FAILURE: Model chọn sai người!")
    
    # 5. Lưu lại để bước 2 dùng
    print(f"\n💾 Saving mining results to {SAVE_FILE}...")
    torch.save({
        'hard_cases': top_hard_cases,
        'img_pids': img_pids,   # Lưu PID để đối chiếu
        'txt_pids': txt_pids,
        # Lưu Indices để sau này load lại ảnh/text gốc nếu cần (optional)
    }, SAVE_FILE)
    
    print("✅ Step 1 Complete! Ready for visualization.")