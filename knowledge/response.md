Author Response: A Hard Negative-Aware Optimization for Multilingual Text-Based Person Search
We sincerely thank all reviewers for their constructive feedback. Below we address each comment individually.
Reviewer 1
R1-C1: Generalization to Other Languages
Reviewer: The paper claims to address "Multilingual" search but primarily evaluates English (CUHK-PEDES) and Vietnamese (3000VnPersonSearch). While these represent high and low-resource scenarios well, testing on additional languages would strengthen the "multilingual" claim.
Response: Thank you for this suggestion. To strengthen our multilingual claim, we have extended our evaluation to include PRW-TPS-CN with Chinese captions, adding a third language and script to our benchmark suite. Our evaluation now covers three typologically diverse languages: Vietnamese (low-resource, Latin-based), English (high-resource, Latin-based), and Chinese (logographic script), demonstrating that mSigLIP-CLoRA generalizes across different linguistic families and resource levels. The results are summarized below.
Results on PRW-TPS-CN (Chinese):
- TPAN: R@1= 21.63% , R@5=42.54% , R@10=52.99% 
- Baseline: R@1= 46.78% , R@5=60.28% , R@10=66.82% , mAP=35.41% , mINP=10.61%
- mSigLIP-CLoRA: R@1=59.35%, R@5=70.58% , R@10=75.48%, mAP=46.44%,, mINP=15.10%

R1-C2: Table VI Analysis
Reviewer: In the text, the authors mention that Table VI compares the baseline with Fixed Weight and Curriculum strategies on the 10% subset of CUHK-PEDES. However, the table itself shows the Curriculum strategy performing significantly worse than the Fixed Weight strategy on Rank@5, Rank@10, mAP, and mINP. While the text acknowledges this trade-off, a slightly deeper analysis of why curriculum learning collapses ranking consistency under data scarcity would be beneficial.
Response: We apologize for this error. The original Table VI mistakenly reported results from 3000VnPersonSearch instead of the 10% CUHK-PEDES subset, which led to an incorrect trade-off analysis in the manuscript. Both the table and the accompanying discussion have been corrected in the revised version. The corrected results (shown below) reveal that the Curriculum strategy achieves the best performance on 4 out of 5 metrics (Rank@1: 57.10%, Rank@10: 84.34%, mAP: 50.90%, mINP: 34.85%), with only a marginal gap on Rank@5 (76.98% vs. 77.18%). This confirms that curriculum learning does not collapse ranking consistency under data scarcity. Rather, the warm-up phase prevents premature hard-negative mining from destabilizing the still-fragile alignment in low-data regimes, yielding a more consistent advantage overall.
Corrected Table VI (10% CUHK-PEDES):
Baseline TBPS-mSigLIP: R@1 = 46.73%, R@5 = 68.65%, R@10 = 77.55%, mAP = 41.75%, mINP = 26.56%.
Ours (Fixed Weight) : R@1 = 56.87%, R@5 = 77.18% (best), R@10 = 84.15%, mAP = 50.70%, mINP = 34.61%.
Ours (Curriculum) : R@1 = 57.10% (best), R@5 = 76.98%, R@10 = 84.34% (best), mAP = 50.90% (best), mINP = 34.85% (best).
Reviewer 2
R2-Q1: Instability at Circle Loss Weight 0.2 Despite Larger Batch Size
Reviewer: In Section IV-D2, you attribute the performance degradation at higher Circle Loss weights to "gradient dominance" from pseudo-hard negatives in small batches. Since LoRA enabled a 3x increase in batch size (from 8 to 24), why does this instability persist at 0.2?
Response: This is an insightful question. Although LoRA enables a 3x batch size increase (8→24), we note three factors that explain the persistent instability at α₅=0.2:
(1) Batch=24 remains small for contrastive learning. Methods such as CLIP operate with batch sizes in the thousands. With only 24 samples per batch, the "hardest negative" changes significantly across mini-batches, leading to high gradient variance.
(2) Circle Loss amplification via γ=128. The scaling factor γ=128 produces large gradient magnitudes. Doubling the weight from 0.1 to 0.2 roughly doubles the Circle Loss gradient contribution, exceeding the regularization capacity of the LoRA low-rank subspace.
(3) False negative amplification. In small batches, some hard negatives are actually false negatives — visually similar persons with different identity labels. Circle Loss assigns these pairs maximal penalty (high α_n). At α₅=0.2, this erroneous gradient signal is amplified further, pushing the optimization away from the correct solution. This is evidenced by Table III, where α₅=0.2 (R@1=49.83%) drops below even the LoRA-only baseline (R@1=49.90%), suggesting the loss is actively harmful at this weight.
More broadly, our framework addresses label noise at the primary loss level through N-ITC (Sigmoid-based noise-robust contrastive loss). The auxiliary Circle Loss operates at a deliberately small weight (α₅=0.1) with curriculum warm-up to limit its exposure to noisy gradients. However, Circle Loss does not explicitly detect false negatives — it treats all cross-identity pairs as true negatives. We identify this as the fundamental root cause of the instability at higher weights: the problem is not batch size alone, but that Circle Loss amplifies erroneous gradients from mislabeled pairs. The curriculum schedule mitigates this by deferring hard-mining until the embedding space has stabilized (epoch 5+), but explicit false negative detection within Circle Loss remains a promising direction for future work.

R2-Q2: Curriculum Schedule Sensitivity
Reviewer: The curriculum strategy starts after a 5-epoch warm-up. How sensitive is the model to the duration of this warm-up (T_warmup) and the ramp-up period (T_ramp)? Would a non-linear (e.g., exponential or cosine) increase in α₅ offer better stability?
Response: The current schedule (T_warmup=5, T_ramp=15, linear ramp) was designed based on two observations:
Warm-up duration (T_warmup=5). LoRA parameters are initialized near zero and require several epochs to establish a stable alignment in the shared embedding space. We set T_warmup=5 to ensure the primary N-ITC loss has converged to a reasonable embedding structure before introducing hard-negative mining. Introducing Circle Loss too early (e.g., T_warmup=0, i.e., the Fixed Weight strategy) yields lower performance across all metrics (Table IV ), confirming the warm-up is beneficial.
Linear ramp over T_ramp=15 epochs. A linear schedule provides a gradual, predictable increase that avoids sudden gradient shocks. Combined with the small target weight (α₅=0.1), the per-epoch increment is only ~0.0067, which we found sufficient for stable integration.
On non-linear alternatives. We chose a linear ramp for simplicity and reproducibility — it introduces no additional hyperparameters beyond T_warmup and T_ramp. A cosine or exponential schedule could potentially offer smoother transitions, particularly in the early ramp phase where the embedding space is most sensitive. We consider this a valuable direction for future investigation, along with a systematic sensitivity analysis of T_warmup and T_ramp.

R2-Q3: Curriculum Overfitting in Low-Data Regime (Table VI)
Reviewer: In Table VI (10% CUHK-PEDES), the Curriculum strategy achieves the highest Rank@1 (57.10%) but results in a significant drop in mAP and mINP compared to the Fixed Weight strategy. Does this suggest that the curriculum mining "overfits" the most distinct features at the expense of a well-calibrated global retrieval space?
Response: This concern was based on incorrect numbers in the original Table VI (see our response to R1-C2). The original table mistakenly reported 3000VnPersonSearch results instead of the 10% CUHK-PEDES subset. After correction, the Curriculum strategy achieves the best performance on 4 out of 5 metrics (R@1 = 57.10%, R@10 = 84.34%, mAP = 50.90%, mINP = 34.85%), with only a marginal difference on R@5 (76.98% vs. 77.18%). This demonstrates that curriculum learning does not overfit to the most distinct features — it preserves global retrieval quality while achieving the strongest peak discrimination. The warm-up phase is in fact beneficial under data scarcity: by deferring Circle Loss until the embedding space is sufficiently aligned, the curriculum avoids premature hard-negative mining that could destabilize the fragile early-stage representations.

R2-Q4: Circle Loss Benefit for Low-Resource vs. High-Resource
Reviewer: While the method shows strong gains on the Vietnamese 3000VnPersonSearch dataset, the improvement on the full CUHK-PEDES dataset is more incremental (Rank@1: 70.76% to 71.85%). Is the primary benefit of Circle Loss more specialized for low-resource languages, or could the margin m=0.25 be suboptimal for high-resource English benchmarks?
Response: We believe the difference in gain reflects the baseline alignment quality rather than a language-specific limitation. On high-resource CUHK-PEDES (13K identities, 80K texts), the primary N-ITC loss already achieves strong alignment, leaving less room for an auxiliary geometric refinement. On low-resource 3000VnPersonSearch (3K identities, 12K texts), the baseline alignment is weaker, so Circle Loss contributes more by enforcing explicit separation between hard negatives.
Additionally, the false negative collision rate differs between the two datasets. With 13K identities in CUHK-PEDES, the probability of sampling a false negative (same person, different PID) within a batch of 24 is substantially lower than with 3K identities in VnPersonSearch. Since Circle Loss amplifies gradients for the hardest negatives — including false negatives — its benefit is naturally larger in scenarios where such noise is more prevalent.
Importantly, the +1.09% improvement on CUHK-PEDES is consistent across metrics and does not degrade any metric, confirming that the method generalizes beyond low-resource settings without causing harm. Regarding the margin m=0.25, we acknowledge this was tuned on VnPersonSearch and may indeed be suboptimal for high-resource benchmarks where the embedding space is already well-separated. A dataset-adaptive margin is a promising direction for future work.

R2-Q5: Qualitative Hard Negative Examples
Reviewer: Figure 3 shows the learned distribution aligning with a spherical geometry. Can you provide examples of specific "hard negative" pairs (e.g., two different people wearing identical uniforms) that the baseline failed to separate but mSigLIP-CLORA successfully distinguished?
Response:

Reviewer 3
R3-C1: Re-run and Correct Table VI
Reviewer: Re-run and correct Table VI.
Response: The original Table VI mistakenly reported results from 3000VnPersonSearch instead of the 10% CUHK-PEDES subset. Both the table and the accompanying analysis have been corrected in the revised manuscript. The corrected results show that the Curriculum strategy achieves the best performance on 4 out of 5 metrics: R@1 = 57.10% (best), R@5 = 76.98%, R@10 = 84.34% (best), mAP = 50.90% (best), mINP = 34.85% (best). See our response to R1-C2 for full details.

R3-C2: Full Fine-Tuning Baseline at Batch=24
Reviewer: Add a full fine-tuning baseline at batch=24 to isolate the Circle Loss contribution from the batch-size effect.
Response:
We have conducted the requested control experiment: full fine-tuning of all 376M parameters at the identical batch configuration as LoRA (batch=24, gradient accumulation=3, effective batch=72) on an RTX 4090 (24 GB), using the same training schedule (60 epochs, cosine LR, AdamW) but without Circle Loss. The results on 3000VnPersonSearch are compared below.
Full Fine-Tuning (batch=24, eff. batch=72, no Circle Loss):
R@1 = 49.18%, R@5 = 76.30%, R@10 = 85.58%, mAP = 54.49%, mINP = 47.87%.
LoRA (batch=24, eff. batch=72, no Circle Loss):
R@1 = 49.90%, R@5 = 77.45%, R@10 = 86.20%, mAP = 55.23%, mINP = 48.65%. (as shown in the paper)
LoRA + Curriculum Circle Loss (ours):
R@1 = 51.30%, R@5 = 78.20%, R@10 = 86.68%, mAP = 56.46%, mINP = 49.89% (even better with seed 2400 as describe in R3-C4)
Key findings:
(1) At the identical batch configuration, full fine-tuning (49.18%) underperforms LoRA (49.90%), conclusively ruling out batch size as the source of improvement. With 376M trainable parameters, full fine-tuning overfits on the small VN3K dataset (3K identities, 12K texts).
(2) The improvement from Circle Loss is +1.40% R@1 (49.90% → 51.30%), measured on top of LoRA at the same batch size, cleanly isolating its contribution.
(3) LoRA's low-rank constraint acts as an implicit regularizer, preventing the overparameterized model from memorizing the limited training set. This three-way comparison confirms that both components — LoRA regularization and Circle Loss hard-negative mining — contribute independently to the final performance.
R3-C3: Report LoRA Rank and Scaling Hyperparameters
Reviewer: Report LoRA rank and scaling hyperparameters.
Response: We apologize for the omission. The LoRA hyperparameters are as follows:
rank r = 32,
scaling alpha = 64 (effective scaling ratio alpha/r = 2.0),
dropout = 0.05, applied to the query, key, value, and output projections (q_proj, k_proj, v_proj, out_proj) of all transformer layers.
This yields 5.9M trainable parameters out of 376M total (1.57%), while the remaining 370M backbone parameters are frozen. These details have been added to Section IV-C (Implementation Details) in the revised manuscript.

R3-C4: Confidence Intervals for Key Comparisons
Reviewer: Add confidence intervals for key comparisons.
Response: We have re-run the main experiment (LoRA + Curriculum Circle Loss) with 3 different random seeds on 3000VnPersonSearch. The results are as follows:
Seed 2307 (reported in paper): R@1 = 51.30%, R@5 = 78.20%, R@10 = 86.68%, mAP = 56.46%, mINP = 49.89%.
Seed 2300: R@1 = 50.98%, R@5 = 78.60%, R@10 = 86.95%, mAP = 57.08%, mINP = 51.22%.
Seed 2400: R@1 = 52.28%, R@5 = 79.55%, R@10 = 88.03%, mAP = 57.32%, mINP = 50.57%.
Mean ± std: R@1 = 51.52 ± 0.68%, R@5 = 78.78 ± 0.69%, R@10 = 87.22 ± 0.71%, mAP = 56.95 ± 0.44%, mINP = 50.56 ± 0.67%.
The low standard deviation (< 0.75% across all metrics) confirms that the improvements are stable and not attributable to random seed selection. These confidence intervals have been added to the revised manuscript.

Reviewer 4
R4-C1: Position Contribution More Carefully
Reviewer: My main concern is that the contribution should be positioned more carefully. The strongest case is in multilingual low-resource adaptation; the method is not uniformly best across all TBPS settings.
Response: We agree with the reviewer and have revised the positioning in the abstract, introduction, and conclusion. The revised manuscript frames the primary contribution as a parameter-efficient optimization framework for multilingual low-resource text-based person search, where the largest gains are observed. We now explicitly acknowledge that improvements on high-resource CUHK-PEDES are more incremental (+1.09% R@1), while the method's strength lies in low-resource settings where baseline alignment is weaker and Circle Loss provides the most benefit. See also R4-C4 below.

R4-C2: Control Experiment for Batch Size Confound
Reviewer: Because LoRA permits materially larger batches than full fine-tuning, some of the gain may come from optimization and batch-size effects rather than the hard-negative loss itself. A stricter control would help isolate attribution.
Response:
We have addressed this concern with a full fine-tuning control at the identical batch configuration (batch=24, accumulation=3, effective batch=72) — see R3-C2 for full details. The three-way comparison on 3000VnPersonSearch yields:
Full FT (eff. batch=72, no Circle): R@1 = 49.18%.
LoRA (eff. batch=72, no Circle): R@1 = 49.90%.
LoRA + Curriculum Circle (ours): R@1 = 51.30%.
At the same effective batch size, full fine-tuning underperforms LoRA (49.18% vs. 49.90%), conclusively ruling out batch size as the source of improvement. The gain decomposes into two independent contributions:
(1) LoRA regularization, which prevents overfitting on the small VN3K dataset by constraining updates to a low-rank subspace;
(2) Curriculum Circle Loss, which adds +1.40% R@1 by explicitly mining hard negatives once the embedding space is sufficiently aligned. Neither component's benefit can be attributed to a batch-size artifact.
R4-C3: Report Trainable Parameter Fractions and Memory Savings
Reviewer: I suggest explicitly reporting trainable parameter fractions and memory savings.
Response: We have added these details to the revised manuscript. The breakdown is as follows:
Full fine-tuning: 376M trainable parameters (100%), ~11 GB VRAM at batch size 8. LoRA: 5.9M trainable parameters (1.57%), ~11 GB VRAM at batch size 24. Both measured on a single NVIDIA RTX 3060 (12 GB).
LoRA reduces trainable parameters by 98.4%, which dramatically reduces optimizer state memory (Adam stores 2 additional copies per trainable parameter). This freed memory is redirected to a 3x larger batch size (8 → 24) on the same hardware. Full fine-tuning at batch=24 exceeds 12 GB and requires gradient accumulation, whereas LoRA runs natively. The larger batch directly enables more effective contrastive learning through a richer negative set per iteration.

R4-C4: Sharpen Multilingual Low-Resource Positioning
Reviewer: Sharpening the positioning so the method is presented as strongest in multilingual low-resource adaptation rather than universally strongest TBPS.
Response: Agreed. As described in R4-C1, we have revised the abstract, introduction, and conclusion to position the method as strongest in multilingual low-resource adaptation. The revised framing emphasizes that Circle Loss provides the greatest benefit when baseline alignment is weak (low-resource), while LoRA enables this on consumer-grade hardware (12 GB GPU). We present the consistent improvement on high-resource CUHK-PEDES as evidence of generalizability, not as the primary claim.

R4-C5: Expand Multilingual Validation
Reviewer: The multilingual generalization claim would be stronger with validation beyond a single low-resource benchmark. Expanding multilingual validation if possible or discussing this limitation more directly.
Response: As described in our response to R1-C1, we have extended our evaluation to include PRW-TPS-CN with Chinese captions, bringing the total to three typologically diverse languages: Vietnamese (low-resource, Latin script), English (high-resource, Latin script), and Chinese (logographic script). This demonstrates that mSigLIP-CLoRA generalizes across different linguistic families and writing systems. We acknowledge that further validation on additional low-resource languages (e.g., Thai, Indonesian) would be valuable and have included this as an explicit limitation and future work direction in the revised manuscript.

