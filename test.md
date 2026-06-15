python3 deployment/scripts/qnn/submit_qaihub_compile_link.py \
    --model artifacts/deployment/runtime/gelu20_w8a16_attn_int8 \
    --device "Dragonwing RB3 Gen 2 Vision Kit" \
    --input-specs '{"image": ((1, 3, 256, 256), "float32")}' \
    --compile-options "--quantize_io" \
    --name msiglip-vision-gelu20-w8a16-attn8-htp \
    --wait \
    --download artifacts/deployment/runtime/gelu20_w8a16_attn_int8/vision_encoder.bin