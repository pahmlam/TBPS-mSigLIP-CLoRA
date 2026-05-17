We thank all reviewers. All corrections and new experiments are reflected in the revised manuscript.

Table VI Error (R1-C2, R2-Q3, R3-C1): The original Table VI mistakenly reported VnPersonSearch results instead of the 10% CUHK-PEDES subset. Corrected results:
Baseline: R@1=46.73%, R@5=68.65%, R@10=77.55%, mAP=41.75%, mINP=26.56%.
Fixed Weight: R@1=56.87%, R@5=77.18%(best), R@10=84.15%, mAP=50.70%, mINP=34.61%.
Curriculum: R@1=57.10%(best), R@5=76.98%, R@10=84.34%(best), mAP=50.90%(best), mINP=34.85%(best).
Curriculum achieves best on 4/5 metrics, confirming it does not collapse ranking consistency under data scarcity.

Multilingual Validation (R1-C1, R4-C5): We added PRW-TPS-CN (Chinese captions), extending evaluation to three typologically diverse languages: Vietnamese (low-resource, Latin), English (high-resource, Latin), Chinese (logographic). Baseline: R@1=--, mSigLIP-CLoRA: R@1=--. Further low-resource languages are acknowledged as future work.

Circle Loss Instability at weight 0.2 (R2-Q1): Three factors explain this: (1) batch=24 remains small for contrastive learning (CLIP uses thousands); (2) gamma=128 amplifies gradients, doubling weight exceeds LoRA's regularization capacity; (3) false negatives receive maximal penalty since Circle Loss treats all cross-identity pairs as true negatives. At weight 0.2, R@1=49.83% drops below the LoRA-only baseline (49.90%), confirming the loss becomes harmful. Our framework handles noise at the primary level (N-ITC); Circle Loss at small weight + curriculum warm-up limits FN exposure. Explicit FN detection is future work.

Curriculum Sensitivity (R2-Q2): T_warmup=5 ensures N-ITC converges before hard-mining (Table IV: Fixed Weight < Curriculum across all metrics). Linear ramp (T_ramp=15, ~0.0067/epoch) avoids gradient shocks. Non-linear schedules are a valid future direction.

Low vs. High Resource Gains (R2-Q4): The gain difference reflects baseline alignment quality, not language limitation. CUHK-PEDES (13K IDs) already has strong alignment and lower FN collision probability than VN3K (3K IDs) at batch=24. The +1.09% on CUHK-PEDES is consistent across all metrics without degradation.

Qualitative Examples (R2-Q5): See attached figure showing cases where baseline fails at Rank@1 but mSigLIP-CLoRA succeeds. [LINK]

Full FT Control at Batch=24 (R3-C2, R4-C2): Full FT of all 376M params at the identical batch config (batch=24, accum=3, eff. batch=72) on RTX 4090: R@1=49.18%. LoRA (same config): R@1=49.90%. Ours (+Circle): R@1=51.30%. Full FT underperforms LoRA at identical batch size, ruling out the batch-size confound. LoRA's low-rank constraint regularizes against overfitting on the small VN3K dataset; Circle Loss adds +1.40% R@1 independently.

LoRA Hyperparameters (R3-C3): r=32, alpha=64 (scaling ratio=2.0), dropout=0.05, applied to q/k/v/out_proj of all transformer layers. 5.9M trainable / 376M total (1.57%). Added to Section IV-C.

Confidence Intervals (R3-C4): 3 seeds on VN3K (LoRA + Curriculum Circle Loss):
Seed 2307 (paper): R@1=51.30%, R@5=78.20%, R@10=86.68%, mAP=56.46%, mINP=49.89%.
Seed 2300: R@1=50.98%, R@5=78.60%, R@10=86.95%, mAP=57.08%, mINP=51.22%.
Seed 2400: R@1=52.28%, R@5=79.55%, R@10=88.03%, mAP=57.32%, mINP=50.57%.
Mean+/-std: R@1=51.52+/-0.68%, R@5=78.78+/-0.69%, R@10=87.22+/-0.71%, mAP=56.95+/-0.44%, mINP=50.56+/-0.67%. All std < 0.75%, confirming stability.

Positioning (R4-C1, R4-C4): Revised abstract, introduction, and conclusion to frame the primary contribution as multilingual low-resource adaptation. CUHK-PEDES improvement is presented as generalizability evidence, not the primary claim.

Memory Savings (R4-C3): Full FT: 376M params (100%), ~11GB VRAM at batch=8. LoRA: 5.9M params (1.57%), ~11GB at batch=24. LoRA reduces trainable params by 98.4%; freed optimizer memory enables 3x batch on same 12GB RTX 3060.
