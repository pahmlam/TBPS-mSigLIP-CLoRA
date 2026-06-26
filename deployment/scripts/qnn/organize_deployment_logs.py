#!/usr/bin/env python3
"""Organize curated deployment logs and result summaries into a canonical archive."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


LOG_JOBS: list[dict[str, Any]] = [
    {
        "job_id": "jgj1wxo1g",
        "source": "artifacts/deployment/logs/internal_floating-point/jgj1wxo1g/jgj1wxo1g.log",
        "target_relpath": "aihub/failures/internal-float/2026-06-08_encoder-blocks-4-11-float__link-fail__add_1003__jgj1wxo1g.log",
        "journal_section": "2026-06-08 - Compile/Link encoder_blocks_4_11_float diagnostic",
        "kind": "link fail",
        "why_relevant": "QDQ local pass but HTP link rejects internal FP tensor add_1003.",
    },
    {
        "job_id": "jgj178xvg",
        "source": "artifacts/deployment/logs/internal_floating-point/jgj178xvg/jgj178xvg.log",
        "target_relpath": "aihub/failures/internal-float/2026-06-09_w8a16-gelu-dequant__link-fail__jgj178xvg.log",
        "journal_section": "2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail",
        "kind": "link fail",
        "why_relevant": "W8A16 local pass rejects floating GELU dequant output during link.",
    },
    {
        "job_id": "j576417rg",
        "source": "artifacts/deployment/logs/internal_floating-point/j576417rg/j576417rg.log",
        "target_relpath": "aihub/failures/internal-float/2026-06-09_w8a16s-gelu-dequant__link-fail__j576417rg.log",
        "journal_section": "2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail",
        "kind": "link fail",
        "why_relevant": "Signed W8A16 variant has same floating GELU dequant failure.",
    },
    {
        "job_id": "jpv4d8rzp",
        "source": "artifacts/deployment/logs/internal_floating-point/jpv4d8rzp/jpv4d8rzp.log",
        "target_relpath": "aihub/failures/internal-float/2026-06-09_htp-fp16-gelu-dequant__link-fail__jpv4d8rzp.log",
        "journal_section": "2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail",
        "kind": "link fail",
        "why_relevant": "HTP FP16/internal activation attempt still leaves floating GELU output.",
    },
    {
        "job_id": "jp389l6z5",
        "source": "artifacts/deployment/logs/internal_floating-point/jp389l6z5/jp389l6z5.log",
        "target_relpath": "aihub/failures/internal-float/2026-06-09_gelu-qint8__link-fail__jp389l6z5.log",
        "journal_section": "2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail",
        "kind": "link fail",
        "why_relevant": "GELU qint8 variant remains in the internal-float link-fail family.",
    },
    {
        "job_id": "j5wx7kmzp",
        "source": "artifacts/deployment/logs/internal_floating-point/j5wx7kmzp/j5wx7kmzp.log",
        "target_relpath": "aihub/failures/internal-float/2026-06-09_gelu-quint8__link-fail__j5wx7kmzp.log",
        "journal_section": "2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail",
        "kind": "link fail",
        "why_relevant": "GELU quint8 variant remains in the internal-float link-fail family.",
    },
    {
        "job_id": "jgl7xd1l5",
        "source": "artifacts/deployment/logs/internal_floating-point/jgl7xd1l5/jgl7xd1l5.log",
        "target_relpath": "aihub/failures/internal-float/2026-06-09_matmul-act-qint8__link-fail__jgl7xd1l5.log",
        "journal_section": "2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail",
        "kind": "link fail",
        "why_relevant": "MatMul activation qint8 variant remains in same internal-float link-fail family.",
    },
    {
        "job_id": "jp0kjyd25",
        "source": "artifacts/deployment/logs/internal_floating-point/jp0kjyd25/jp0kjyd25.log",
        "target_relpath": "aihub/failures/internal-float/2026-06-09_blocks-0-11-int16__link-fail__add_103_updated__jp0kjyd25.log",
        "journal_section": "2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail",
        "kind": "link fail",
        "why_relevant": "blocks_0_11 int16 variant rejects remaining floating tensor add_103_updated.",
    },
    {
        "job_id": "jgzwm63xg",
        "source": "artifacts/deployment/logs/internal_floating-point/jgzwm63xg/jgzwm63xg.log",
        "target_relpath": "aihub/failures/htp-v73-required/2026-06-09_w8a16-quantize-full-int16__link-fail__matmul-v73-required__jgzwm63xg.log",
        "journal_section": "2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail",
        "kind": "link fail",
        "why_relevant": "Context conversion fails because activation matmul pattern needs newer HTP.",
    },
    {
        "job_id": "jpr90k37p",
        "source": "artifacts/deployment/logs/internal_floating-point/jpr90k37p/jpr90k37p.log",
        "target_relpath": "aihub/failures/htp-v73-required/2026-06-09_gelu-float-for-requant-int16__link-fail__matmul-v73-required__jpr90k37p.log",
        "journal_section": "2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail",
        "kind": "link fail",
        "why_relevant": "GELU float-for-requant + int16 exits during context conversion.",
    },
    {
        "job_id": "j576q80rg",
        "source": "artifacts/deployment/logs/internal_floating-point/j576q80rg/j576q80rg.log",
        "target_relpath": "aihub/failures/htp-v73-required/2026-06-14_w8a16-attention-matmul__link-fail__j576q80rg.log",
        "journal_section": "2026-06-14 - W8A16 Link Fail Trên HTP v68",
        "kind": "link fail",
        "why_relevant": "Native W8A16 QAT path has high fidelity but HTP v68 link fails.",
    },
    {
        "job_id": "jpxmw8kjg",
        "source": "artifacts/deployment/logs/internal_floating-point/jpxmw8kjg/jpxmw8kjg.log",
        "target_relpath": "aihub/failures/htp-v73-required/2026-06-15_mixed-a16-layernorm__link-fail__jpxmw8kjg.log",
        "journal_section": "2026-06-14 - mixed-int attention int8 + rest int16",
        "kind": "link fail",
        "why_relevant": "Mixed A16 path fails around LayerNorm, supporting all-W8A8 constraint.",
    },
    {
        "job_id": "j56re0vy5",
        "source": "artifacts/deployment/logs/internal_floating-point/j56re0vy5/j56re0vy5.log",
        "target_relpath": "aihub/failures/text-mask/2026-06-19_full-text-f32mask__link-fail__cast-output-0__j56re0vy5.log",
        "journal_section": "2026-06-19 - Text f32-mask link fail",
        "kind": "link fail",
        "why_relevant": "Full text graph rejects internal /text_model/Cast_output_0_updated mask tensor.",
    },
    {
        "job_id": "jglo3qz8g",
        "source": "artifacts/deployment/logs/internal_floating-point/jglo3qz8g/jglo3qz8g.log",
        "target_relpath": "aihub/failures/text-mask/2026-06-20_split-text-f32mask__link-fail__cast-output-0__jglo3qz8g.log",
        "journal_section": "2026-06-20 - B2 split-text link FAIL `/Cast_output_0_updated`",
        "kind": "link fail",
        "why_relevant": "Split-text graph still contains mask Cast float island.",
    },
    {
        "job_id": "j56rn6ry5",
        "source": "artifacts/deployment/logs/internal_floating-point/j56rn6ry5/j56rn6ry5.log",
        "target_relpath": "aihub/failures/text-mask/2026-06-20_split-text-mask-expand__link-fail__expand-coef__j56rn6ry5.log",
        "journal_section": "2026-06-20 - split-text re-submit after Cast fix",
        "kind": "link fail",
        "why_relevant": "Cast island removed, but materialized Expand mask coefficient is rejected.",
    },
    {
        "job_id": "jp2j211q5",
        "source": "artifacts/deployment/logs/internal_floating-point/jp2j211q5/jp2j211q5.log",
        "target_relpath": "aihub/successes/2026-06-15_rotation-w8a8-first-link-pass__jp2j211q5.log",
        "journal_section": "2026-06-14 - W8A8 M3 link success",
        "kind": "link success control",
        "why_relevant": "All-W8A8 link success after avoiding internal floating islands.",
    },
    {
        "job_id": "jgoovyj4g",
        "target_relpath": "aihub/successes/2026-06-22_vision-v9-qdq-final__jgoovyj4g.log",
        "journal_section": "2026-06-22 - Final vision v9 QDQ",
        "kind": "quantize success",
        "why_relevant": "Final refined vision v9 QDQ used by current proxy and board recipe.",
    },
    {
        "job_id": "jpe8wr61p",
        "target_relpath": "aihub/successes/2026-06-22_text-i32-f32mask-q dq-final__jpe8wr61p.log".replace("q dq", "qdq"),
        "journal_section": "2026-06-22 - Final text QDQ proxy",
        "kind": "quantize success",
        "why_relevant": "Final text QDQ proxy paired with vision v9 in both-INT8 proxy.",
    },
    {
        "job_id": "jp24xxn65",
        "target_relpath": "aihub/successes/2026-06-18_vision-v8-learned-rotation-q dq-pass__jp24xxn65.log".replace("q dq", "qdq"),
        "journal_section": "2026-06-18 - QAT v8 learned rotation",
        "kind": "quantize success",
        "why_relevant": "First learned-rotation QDQ pass; clean ablation learned vs random.",
    },
    {
        "job_id": "jp17y648p",
        "target_relpath": "aihub/successes/2026-06-19_text-v8-finite-mask-q dq-pass__jp17y648p.log".replace("q dq", "qdq"),
        "journal_section": "2026-06-19 - C1 Off-board both-INT8 retrieval PASS",
        "kind": "quantize success",
        "why_relevant": "Text learned-rotation finite-mask QDQ pass used in C1 proxy.",
    },
    {
        "job_id": "jp383qmn5",
        "target_relpath": "aihub/ablations/vision_qat-v3__t2i48.20_cos0.9353__jp383qmn5.log",
        "journal_section": "2026-06-15 - QAT v3",
        "kind": "quantize ablation",
        "why_relevant": "EMA observer first stable INT8 QAT result.",
    },
    {
        "job_id": "jgd09l96p",
        "target_relpath": "aihub/ablations/vision_qat-v4__t2i48.50_cos0.9364__jgd09l96p.log",
        "journal_section": "2026-06-15 - QAT v4",
        "kind": "quantize ablation",
        "why_relevant": "Head fake-quant and board-verified binary milestone.",
    },
    {
        "job_id": "jpxm2w0lg",
        "target_relpath": "aihub/ablations/vision_qat-v5__t2i49.25_cos0.9437__jpxm2w0lg.log",
        "journal_section": "2026-06-15 - QAT v5",
        "kind": "quantize ablation",
        "why_relevant": "Per-linear fake-quant improves worst-sample cosine.",
    },
    {
        "job_id": "j57krdwvp",
        "target_relpath": "aihub/ablations/vision_qat-v6__t2i49.30_cos0.9491__j57krdwvp.log",
        "journal_section": "2026-06-15 - QAT v6",
        "kind": "quantize ablation",
        "why_relevant": "Random-rotation coverage ceiling before learned rotation.",
    },
    {
        "job_id": "jpve62jmg",
        "target_relpath": "aihub/ablations/vision_qat-v7-regress__t2i48.38_cos0.9485__jpve62jmg.log",
        "journal_section": "2026-06-15 - QAT v7",
        "kind": "quantize ablation",
        "why_relevant": "Cosine LR/lr 2e-5 regression.",
    },
    {
        "job_id": "jgjoly0ep",
        "target_relpath": "aihub/ablations/split-text-calib500__md5-baseline__jgjoly0ep.log",
        "journal_section": "2026-06-22 - split-text calibration md5",
        "kind": "quantize ablation",
        "why_relevant": "Split-text calibration 500 md5 baseline.",
    },
    {
        "job_id": "jp24mdmq5",
        "target_relpath": "aihub/ablations/split-text-calib2000__md5-identical__jp24mdmq5.log",
        "journal_section": "2026-06-22 - split-text calibration md5",
        "kind": "quantize ablation",
        "why_relevant": "Split-text calibration 2000 produced byte-identical QDQ model.",
    },
]


RESULT_MOVES: list[dict[str, Any]] = [
    {
        "source": "artifacts/deployment/qnn_runs/both_int8_board_r1.json",
        "target_relpath": "results/board/final_both-int8_v9-splittext__t2i50.35_i2t54.20.json",
        "type": "board_result",
        "metrics": {"t2i_r1": 50.35, "i2t_r1": 54.20},
    },
    {
        "source": "artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v9_gallery_2000/board_vision_r1.json",
        "target_relpath": "results/board/vision_v9_board-isolation__t2i50.35_i2t54.55.json",
        "type": "board_result",
        "metrics": {"t2i_r1": 50.35, "i2t_r1": 54.55},
    },
    {
        "source": "artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_f32mask/board_text_r1.json",
        "target_relpath": "results/board/text_split-board-isolation__t2i51.30_i2t54.80.json",
        "type": "board_result",
        "metrics": {"t2i_r1": 51.30, "i2t_r1": 54.80},
    },
    {
        "source": "artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000/board_vision_r1.json",
        "target_relpath": "results/board/vision_v8_board-isolation__t2i50.20_i2t54.50.json",
        "type": "board_result",
        "metrics": {"t2i_r1": 50.20, "i2t_r1": 54.50},
    },
    {
        "source": "artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_i32_f32mask/qnn_vs_pytorch_summary.json",
        "target_relpath": "results/board/text_fullgraph-i32__board-fidelity-fail__cos0.126_inputids-ignored.json",
        "type": "board_fidelity_failure",
        "metrics": {"cosine_l2_mean": 0.12666028},
    },
    {
        "source": "artifacts/deployment/runtime/rotated_w8a8_learned_qat_v9/both_int8_qdq_r1.json",
        "target_relpath": "results/qdq/both-int8_v9_proxy__t2i50.63_i2t53.90.json",
        "type": "qdq_result",
        "metrics": {"t2i_r1": 50.63, "i2t_r1": 53.90},
    },
    {
        "source": "artifacts/deployment/runtime/both_int8/both_int8_r1.json",
        "target_relpath": "results/qdq/both-int8_v8_proxy__t2i50.25_i2t52.95.json",
        "type": "qdq_result",
        "metrics": {"t2i_r1": 50.25, "i2t_r1": 52.95},
    },
    {
        "source": "artifacts/deployment/runtime/rotated_w8a8_learned_qat_v9/retrieval_r1.json",
        "target_relpath": "results/qdq/vision_v9_qdq-isolation__t2i50.98_i2t54.20.json",
        "type": "qdq_result",
        "metrics": {"t2i_r1": 50.98, "i2t_r1": 54.20},
    },
    {
        "source": "artifacts/deployment/runtime/rotated_w8a8_v2/retrieval_r1.json",
        "target_relpath": "results/qdq/vision_rotation-only__t2i45.42_cos0.8975.json",
        "type": "qdq_result",
        "metrics": {"t2i_r1": 45.42, "cosine_l2_mean": 0.8975},
    },
    {
        "source": "artifacts/deployment/runtime/rotated_w8a8_qat_v3/retrieval_r1.json",
        "target_relpath": "results/qdq/vision_qat-v3__t2i48.20_cos0.9353.json",
        "type": "qdq_result",
        "metrics": {"t2i_r1": 48.20, "cosine_l2_mean": 0.9353},
    },
    {
        "source": "artifacts/deployment/runtime/rotated_w8a8_qat_v4/retrieval_r1.json",
        "target_relpath": "results/qdq/vision_qat-v4__t2i48.50_cos0.9364.json",
        "type": "qdq_result",
        "metrics": {"t2i_r1": 48.50, "cosine_l2_mean": 0.9364},
    },
    {
        "source": "artifacts/deployment/runtime/rotated_w8a8_qat_v5/retrieval_r1.json",
        "target_relpath": "results/qdq/vision_qat-v5__t2i49.25_cos0.9437.json",
        "type": "qdq_result",
        "metrics": {"t2i_r1": 49.25, "cosine_l2_mean": 0.9437},
    },
    {
        "source": "artifacts/deployment/runtime/rotated_w8a8_qat_v6/retrieval_r1.json",
        "target_relpath": "results/qdq/vision_qat-v6__t2i49.30_cos0.9491.json",
        "type": "qdq_result",
        "metrics": {"t2i_r1": 49.30, "cosine_l2_mean": 0.9491},
    },
    {
        "source": "artifacts/deployment/runtime/rotated_w8a8_qat_v7/retrieval_r1.json",
        "target_relpath": "results/qdq/vision_qat-v7-regress__t2i48.38_cos0.9485.json",
        "type": "qdq_result",
        "metrics": {"t2i_r1": 48.38, "cosine_l2_mean": 0.9485},
    },
    {
        "source": "artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/text_qdq_fid.json",
        "target_relpath": "results/qdq/text_finite-mask_fidelity__cos0.9949_min0.9912.json",
        "type": "qdq_fidelity",
        "metrics": {"cosine_l2_mean": 0.9949, "cosine_l2_min": 0.9912},
    },
    {
        "source": "artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/text_isolation_r1.json",
        "target_relpath": "results/qdq/text_finite-mask_isolation__t2i51.65_i2t55.55.json",
        "type": "qdq_result",
        "metrics": {"t2i_r1": 51.65, "i2t_r1": 55.55},
    },
    {
        "source": "artifacts/deployment/runtime/text_w8a8_learned_qat_v8_finite_mask/attention_qdq_scales.json",
        "target_relpath": "results/qdq/text_attention-mask-scales__max0.3523.json",
        "type": "qdq_diagnostic",
        "metrics": {"max_attention_qdq_scale": 0.3523},
    },
    {
        "source": "artifacts/deployment/runtime/diag/text_outliers/summary.json",
        "target_relpath": "diagnostics/text_outliers_before-rotation__residual-conc404x.json",
        "type": "diagnostic",
        "metrics": {"residual_concentration_max": 404},
    },
    {
        "source": "artifacts/deployment/runtime/diag/text_outliers_rotated/summary.json",
        "target_relpath": "diagnostics/text_outliers_after-rotation__residual-conc5.3x.json",
        "type": "diagnostic",
        "metrics": {"residual_concentration_max": 5.3},
    },
    {
        "source": "artifacts/deployment/runtime/qnn_native/env_audit.json",
        "target_relpath": "diagnostics/qnn_native_env_audit.json",
        "type": "diagnostic",
        "metrics": {},
    },
]


def _runtime_moves(source_dir: str, target_dir: str, files: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "source": f"artifacts/deployment/qnn_runs/{source_dir}/{name}",
            "target_relpath": f"runtime/board/{target_dir}/{name}",
            "type": "board_runtime",
        }
        for name in files
    ]


QNN_RUN_MOVES: list[dict[str, Any]] = [
    {
        "source": "artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/qnn_vs_pytorch_summary.json",
        "target_relpath": "results/board/vision_v8_board-fidelity-smoke10__cos0.9585_min0.9400.json",
        "type": "board_fidelity",
        "metrics": {"cosine_l2_mean": 0.9585, "cosine_l2_min": 0.9400},
    },
    {
        "source": "artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8/qnn_vs_pytorch.csv",
        "target_relpath": "results/board/per-sample/vision_v8_board-fidelity-smoke10__qnn-vs-pytorch.csv",
        "type": "board_fidelity_csv",
        "metrics": {"num_samples": 10},
    },
    {
        "source": "artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000/qnn_vs_pytorch_summary.json",
        "target_relpath": "results/board/vision_v8_gallery-fidelity__cos0.9562_min0.8840.json",
        "type": "board_fidelity",
        "metrics": {"cosine_l2_mean": 0.9562, "cosine_l2_min": 0.8840},
    },
    {
        "source": "artifacts/deployment/qnn_runs/rotated_w8a8_learned_qat_v8_gallery_2000/qnn_vs_pytorch.csv",
        "target_relpath": "results/board/per-sample/vision_v8_gallery-fidelity__qnn-vs-pytorch.csv",
        "type": "board_fidelity_csv",
        "metrics": {"num_samples": 2000},
    },
    {
        "source": "artifacts/deployment/qnn_runs/text_w8a8_learned_qat_v8_i32_f32mask/qnn_vs_pytorch.csv",
        "target_relpath": "results/board/per-sample/text_fullgraph-i32__board-fidelity-fail__qnn-vs-pytorch.csv",
        "type": "board_fidelity_failure_csv",
        "metrics": {"num_samples": 10},
    },
]

QNN_RUN_MOVES.extend(
    _runtime_moves(
        "rotated_w8a8_learned_qat_v9_gallery_2000",
        "vision_v9_final-gallery2000__32.54ms_24.29fps",
        ["execution_metadata.yaml", "profile.txt", "qnn-profiling-data_0.log"],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "onboard_text",
        "text_split_onboard-query4000-final__7.87ms_74.75qps",
        [
            "execution_metadata.yaml",
            "profile.txt",
            "qnn-profiling-data_0.log",
            "qnn-profiling-data_1.log",
            "qnn-profiling-data_2.log",
        ],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "rotated_w8a8_learned_qat_v8",
        "vision_v8_smoke10__33.05ms_22.77fps",
        [
            "execution_metadata.yaml",
            "profile.txt",
            "qnn-profiling-data_0.log",
            "qnn-profiling-data_1.log",
            "qnn-profiling-data_2.log",
            "qnn-profiling-data_3.log",
            "qnn-profiling-data_4.log",
        ],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "rotated_w8a8_learned_qat_v8_gallery_2000",
        "vision_v8_gallery2000",
        ["execution_metadata.yaml", "qnn-profiling-data_0.log"],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "rotated_w8a8_learned_qat_v9",
        "vision_v9_gallery2000-preprofile",
        ["execution_metadata.yaml", "qnn-profiling-data_0.log"],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "split_text_query_full",
        "text_split_hostembeds-query4000",
        ["execution_metadata.yaml", "profile.txt", "qnn-profiling-data_0.log"],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "split_text_query_4000",
        "text_split_hostembeds-query2000",
        ["execution_metadata.yaml", "qnn-profiling-data_0.log"],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "split_text_w8a8",
        "text_split_smoke10-real-embeds",
        ["execution_metadata.yaml", "qnn-profiling-data_0.log", "qnn-profiling-data_1.log"],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "split_text_w8a8_zero",
        "text_split_smoke10-zero-embeds-control",
        ["execution_metadata.yaml", "qnn-profiling-data_0.log", "qnn-profiling-data_1.log"],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "text_w8a8_learned_qat_v8_i32_f32mask",
        "text_fullgraph_i32-realids-fail",
        [
            "execution_metadata.yaml",
            "qnn-profiling-data_0.log",
            "qnn-profiling-data_1.log",
            "qnn-profiling-data_2.log",
            "qnn-profiling-data_3.log",
            "qnn-profiling-data_4.log",
        ],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "text_w8a8_learned_qat_v8_i32_zero_ids",
        "text_fullgraph_i32-zeroids-control",
        ["execution_metadata.yaml", "qnn-profiling-data_0.log"],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "text_altreal",
        "text_fullgraph_i32-altreal-mask-control",
        ["execution_metadata.yaml", "qnn-profiling-data_0.log", "qnn-profiling-data_1.log"],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "vision_ram_one",
        "vision_v9_ram-one",
        ["execution_metadata.yaml", "qnn-profiling-data_0.log"],
    )
)
QNN_RUN_MOVES.extend(
    _runtime_moves(
        "text_ram_one",
        "text_split_ram-one",
        ["execution_metadata.yaml", "qnn-profiling-data_0.log", "qnn-profiling-data_1.log"],
    )
)


LEGACY_MOVES: list[dict[str, Any]] = [
    {
        "source": "artifacts/deployment/logs/internal_floating-point/download_manifest.json",
        "target_relpath": "aihub/legacy/internal-floating-point_download_manifest.json",
        "type": "legacy_manifest",
    },
    {
        "source": "artifacts/deployment/logs/internal_floating-point/README.md",
        "target_relpath": "aihub/legacy/internal-floating-point_README.md",
        "type": "legacy_readme",
    },
]


README_TEXT = """# Deployment Evidence Archive

This directory is the canonical evidence archive for the mSigLIP RB3/QNN
deployment work. It keeps small, curated summaries and AI Hub logs under
semantic names, while large generated artifacts remain in their working
locations.

## Layout

| Path | Contents |
|---|---|
| `aihub/` | Curated Qualcomm AI Hub job logs, named by result/failure mode with job ID retained for traceability. |
| `results/board/` | Board retrieval and board fidelity JSON summaries. |
| `results/board/per-sample/` | Small per-sample board fidelity CSV tables. |
| `results/qdq/` | Off-board QDQ proxy retrieval/fidelity JSON summaries. |
| `runtime/board/` | Board `qnn-net-run` execution metadata, generated profiles, and profiling logs. |
| `diagnostics/` | Small diagnostic JSON summaries for environment, mask, and activation-outlier checks. |
| `manifest.json` | Provenance map from original source paths to canonical evidence paths. |

Large model files, QNN context binaries, ONNX/QDQ model directories, raw inputs,
and board `Result_*` outputs are intentionally not stored here.
"""


def _move_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in LOG_JOBS:
        source = item.get("source")
        if source:
            specs.append({**item, "type": "aihub_log"})
    specs.extend(RESULT_MOVES)
    specs.extend(QNN_RUN_MOVES)
    specs.extend(LEGACY_MOVES)
    return specs


def _write_text(path: Path, text: str, apply: bool) -> None:
    print(f"WRITE {path}")
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _write_json(path: Path, data: Any, apply: bool) -> None:
    print(f"WRITE {path}")
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _same_file_content(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


def _move_file(repo: Path, logs_root: Path, spec: dict[str, Any], apply: bool, overwrite: bool) -> dict[str, Any]:
    source = repo / spec["source"]
    target = logs_root / spec["target_relpath"]
    record = {
        "source_path": str(source.relative_to(repo)),
        "canonical_path": str(target.relative_to(repo)),
        "type": spec.get("type", "evidence"),
        "job_id": spec.get("job_id"),
        "metrics": spec.get("metrics", {}),
        "status": "planned",
    }
    if not source.exists():
        if target.exists():
            record["status"] = "already_canonical"
            print(f"HAVE  {target}")
            return record
        record["status"] = "source_missing"
        print(f"MISS  {source}")
        return record

    if target.exists():
        if _same_file_content(source, target):
            print(f"DEDUP {source} -> {target}")
            record["status"] = "deduped_existing_target"
            if apply:
                source.unlink()
            return record
        if not overwrite:
            raise FileExistsError(f"Target exists and differs: {target}")
        print(f"OVERWRITE {target}")

    print(f"MOVE  {source} -> {target}")
    record["status"] = "moved"
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and overwrite:
            target.unlink()
        shutil.move(str(source), str(target))
    return record


def _cleanup_empty_dirs(path: Path, stop_at: Path, apply: bool) -> None:
    current = path
    while current != stop_at and current.exists():
        try:
            next(current.iterdir())
            return
        except StopIteration:
            print(f"RMDIR {current}")
            if apply:
                current.rmdir()
            current = current.parent


def _remove_ds_store(root: Path, apply: bool) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob(".DS_Store")):
        print(f"REMOVE {path}")
        if apply:
            path.unlink()


def _remove_qnn_run_profile_aliases(root: Path, apply: bool) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("qnn-profiling-data.log")):
        if not path.is_symlink():
            continue
        print(f"REMOVE {path}")
        if apply:
            path.unlink()


def _cleanup_empty_tree(root: Path, apply: bool) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            next(path.iterdir())
        except StopIteration:
            print(f"RMDIR {path}")
            if apply:
                path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=_repo_root() / "artifacts/deployment/logs")
    parser.add_argument("--apply", action="store_true", help="Actually move files; default is dry-run.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = _repo_root()
    logs_root = args.logs_root.expanduser().resolve()
    qnn_runs_root = repo / "artifacts/deployment/qnn_runs"

    print("Mode:", "apply" if args.apply else "dry-run")
    print("Logs root:", logs_root)

    records = []
    for spec in _move_specs():
        records.append(_move_file(repo, logs_root, spec, args.apply, args.overwrite))

    manifest = {
        "description": "Canonical deployment evidence archive manifest.",
        "generated_by": "deployment/scripts/qnn/organize_deployment_logs.py",
        "records": records,
    }
    _write_json(logs_root / "manifest.json", manifest, args.apply)
    _write_text(logs_root / "README.md", README_TEXT, args.apply)

    jobs_for_download = [
        {k: v for k, v in job.items() if k != "source"}
        for job in LOG_JOBS
    ]
    _write_json(logs_root / "aihub/curated_jobs.json", {"jobs": jobs_for_download}, args.apply)

    _remove_ds_store(logs_root, args.apply)
    _remove_ds_store(qnn_runs_root, args.apply)
    _remove_qnn_run_profile_aliases(qnn_runs_root, args.apply)
    _cleanup_empty_dirs(logs_root / "internal_floating-point", logs_root, args.apply)
    _cleanup_empty_tree(qnn_runs_root, args.apply)

    moved = sum(1 for record in records if record["status"] == "moved")
    missing = sum(1 for record in records if record["status"] == "source_missing")
    canonical = sum(1 for record in records if record["status"] == "already_canonical")
    print(f"Summary: moved={moved}, already_canonical={canonical}, source_missing={missing}, records={len(records)}")


if __name__ == "__main__":
    main()
