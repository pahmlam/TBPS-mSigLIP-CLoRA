# Internal Floating-Point AI Hub Logs

Downloaded via `qai_hub.get_job(job_id).download_job_logs(...)` into per-job subdirectories.

| Job | Kind | Logs | Key evidence | Journal context |
|---|---|---:|---|---|
| `jgj1wxo1g` | link fail | OK | [2026-06-06T17:17:42.459+00:00] [INFO] Tensor 'add_1003' has a floating-point type which is not supported by the targeted device. | 2026-06-08 - Compile/Link encoder_blocks_4_11_float diagnostic |
| `jgj178xvg` | link fail | OK | [2026-06-09T01:46:55.842+00:00] [INFO] Tensor 'gelu_10_DequantizeLinear_Output' has a floating-point type which is not supported by the targeted device. | 2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail |
| `j576417rg` | link fail | OK | [2026-06-09T02:27:58.262+00:00] [INFO] Tensor 'gelu_10_DequantizeLinear_Output' has a floating-point type which is not supported by the targeted device. | 2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail |
| `jgzwm63xg` | link fail | OK | [2026-06-09T04:06:04.551+00:00] [INFO] [ ERROR ]  <E> [4294967295] has incorrect Value 68, expected >= 73. | 2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail |
| `jpv4d8rzp` | link fail | OK | [2026-06-09T04:22:59.699+00:00] [INFO] Tensor 'gelu_10_DequantizeLinear_Output' has a floating-point type which is not supported by the targeted device. | 2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail |
| `jp389l6z5` | link fail | OK | [2026-06-09T02:47:57.262+00:00] [INFO] Tensor 'gelu_10_DequantizeLinear_Output' has a floating-point type which is not supported by the targeted device. | 2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail |
| `j5wx7kmzp` | link fail | OK | [2026-06-09T04:58:53.995+00:00] [INFO] Tensor 'gelu_10_DequantizeLinear_Output' has a floating-point type which is not supported by the targeted device. | 2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail |
| `jgl7xd1l5` | link fail | OK | [2026-06-09T05:13:59.627+00:00] [INFO] Tensor 'gelu_10_DequantizeLinear_Output' has a floating-point type which is not supported by the targeted device. | 2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail |
| `jpr90k37p` | link fail | OK | [2026-06-09T07:24:54.810+00:00] [INFO] [ ERROR ]  <E> [4294967295] has incorrect Value 68, expected >= 73. | 2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail |
| `jp0kjyd25` | link fail | OK | [2026-06-09T08:12:59.218+00:00] [INFO] Tensor 'add_103_updated' has a floating-point type which is not supported by the targeted device. | 2026-06-09 - ORT W8A16 Pass Local, QNN Link Fail |
| `j576q80rg` | link fail | OK | [2026-06-14T13:02:59.666+00:00] [INFO] [ ERROR ]  <E> [4294967295] has incorrect Value 68, expected >= 73. | 2026-06-14 - W8A16 Link Fail Trên HTP v68 |
| `jpxmw8kjg` | link fail | OK | [2026-06-15T00:41:19.464+00:00] [INFO] [ ERROR ]  <E> [4294967295] has incorrect Value 68, expected >= 73. | 2026-06-14 - mixed-int attention int8 + rest int16 |
| `j56re0vy5` | link fail | OK | [2026-06-19T08:19:27.679+00:00] [INFO] Tensor '/text_model/Cast_output_0_updated' has a floating-point type which is not supported by the targeted device. | 2026-06-19 - Text f32-mask link fail |
| `jglo3qz8g` | link fail | OK | [2026-06-20T11:49:38.830+00:00] [INFO] Tensor '/Cast_output_0_updated' has a floating-point type which is not supported by the targeted device. | 2026-06-20 - B2 split-text link FAIL `/Cast_output_0_updated` |
| `j56rn6ry5` | link fail | OK | [2026-06-20T12:21:50.022+00:00] [INFO] Tensor '/Expand_coef' has a floating-point type which is not supported by the targeted device. | 2026-06-20 - split-text re-submit after Cast fix |
| `jp2j211q5` | link success control | OK |  | 2026-06-14 - W8A8 M3 link success |

See `download_manifest.json` for full per-job status, URL, notes, and evidence-line excerpts.
