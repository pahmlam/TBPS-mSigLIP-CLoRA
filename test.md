python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py --model artifacts/deployment/exports/exported_model_rotated_qat_v4/vision_onnx --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 --wait --download artifacts/deployment/bin/w8a8_rotated_qat_v4/vision_encoder.bin


QAIRT=/opt/qcom/qairt/2.45.40.260406; export QNN_LIB=$QAIRT/lib/aarch64-ubuntu-gcc9.4 LD_LIBRARY_PATH=$QAIRT/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH ADSP_LIBRARY_PATH=$QAIRT/lib/hexagon-v68/unsigned; cd /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_inputs/vn3k_test_10 && qnn-net-run --backend $QNN_LIB/libQnnHtp.so --retrieve_context /home/ubuntu/sigm/Lam/artifacts/deployment/bin/w8a8_rotated_qat_v4/vision_encoder.bin --config_file /home/ubuntu/sigm/Lam/deployment/config/qnn/htp_config_245.json --input_list input_list.txt --output_dir /home/ubuntu/sigm/Lam/artifacts/deployment/qnn_runs/rotated_w8a8_qat_v4 --profiling_level basic --perf_profile high_performance

python3 deployment/scripts/qnn/compare_qnn_with_pytorch.py --qnn-output-dir artifacts/deployment/qnn_runs/rotated_w8a8_qat_v4 --model-dir artifacts/deployment/exports/exported_model --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32 --json artifacts/deployment/qnn_runs/rotated_w8a8_qat_v4/qnn_vs_pytorch_summary.json --csv artifacts/deployment/qnn_runs/rotated_w8a8_qat_v4/qnn_vs_pytorch.csv

python3 deployment/scripts/qnn/submit_qaihub_quantize_compile.py --model artifacts/deployment/exports/exported_model_rotated_qat_v5/vision_onnx --calibration-data d7jzjy1m2 --weights-dtype int8 --activations-dtype int8 --quantize-only --wait --download-quantized artifacts/deployment/runtime/rotated_w8a8_qat_v5/job_qdq_onnx/model.onnx

python3 deployment/scripts/qnn/compare_onnx_with_pytorch.py --onnx-model artifacts/deployment/runtime/rotated_w8a8_qat_v5/job_jpxm2w0lg_qdq_onnx --model-dir artifacts/deployment/exports/exported_model --input-dir artifacts/deployment/qnn_inputs/vn3k_test_10 --precision fp32 --json artifacts/deployment/runtime/rotated_w8a8_qat_v5/qdq_vs_pytorch_summary.json --csv artifacts/deployment/runtime/rotated_w8a8_qat_v5/qdq_vs_pytorch.csv

python3 deployment/scripts/qnn/eval_retrieval_quantized_vision.py --qdq-onnx artifacts/deployment/runtime/rotated_w8a8_qat_v5/job_jpxm2w0lg_qdq_onnx --json artifacts/deployment/runtime/rotated_w8a8_qat_v5/retrieval_r1.json

PYTHONUNBUFFERED=1 python deployment/scripts/qnn/train_vision_quant_robust.py \
    --model-dir artifacts/deployment/exports/exported_model_rotated \
    --train-input-dir artifacts/deployment/qnn_inputs/vn3k_train_all_4302 \
    --val-input-dir artifacts/deployment/qnn_inputs/vn3k_test_100 \
    --output-dir artifacts/deployment/exports/exported_model_rotated_qat_v6 \
    --device cuda --batch-size 24 --epochs 15 --lr 1e-5 \
    --fake-quant-observer ema --quant-head --quant-linears --quant-attention \
    --start-layer 0 --end-layer 11 --num-workers 4