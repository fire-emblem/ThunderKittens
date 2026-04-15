# Phase-0 Baseline Summary

Total rows: 39

## 0x0x0
- tk_local_layoutc_4096x4096x4096_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed

## 1664x1024x16384
- cute_shape_selected_best_1664x1024x16384_bf16: cute_tk_layoutc_tile128x128x128_stage4 | tflops_avg=152.74534533333335 | err_max=0.0 | status=ok
- cute_swizzled_tn_1664x1024x16384_bf16: cute_tk_swizzled_tn_tile128x128x128_stage4 | tflops_avg=150.26802733333332 | err_max=0.0 | status=ok
- cute_layoutc_1664x1024x16384_bf16: cute_tk_layoutc_tile128x128x128_stage4 | tflops_avg=152.84649666666667 | err_max=0.0 | status=ok

## 2048x2048x2048
- cute_shape_selected_best_2048cube_bf16: cute_tk_layoutc_tile128x128x128_stage4 | tflops_avg=98.99998933333333 | err_max=0.0 | status=ok
- cute_square_tt_2048x2048x2048_bf16: cute_tk_square_tt_tile256x256x64_stage4 | tflops_avg=60.053300666666665 | err_max=3.0 | status=ok
- cute_swizzled_tn_2048cube_bf16: cute_tk_swizzled_tn_tile128x128x128_stage4 | tflops_avg=96.776207 | err_max=0.0 | status=ok
- cute_layoutc_2048x2048x2048_bf16: cute_tk_layoutc_tile128x128x128_stage4 | tflops_avg=99.018182 | err_max=0.0 | status=ok
- tk_local_layoutc_2048x2048x2048_bf16: tk_local_bf16_layoutc_128x128x128_stage4 | tflops_avg=100.234045 | err_max=0.0 | status=ok

## 256x256x64
- cute_square_tt_256x256x64_bf16: cute_tk_square_tt_tile256x256x64_stage4 | tflops_avg=0.05724666666666667 | err_max=0.0 | status=ok

## 3584x128x18944
- cute_shape_selected_best_3584x128x18944_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_swizzled_tn_3584x128x18944_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_continuousc_reusea_3584x128x18944: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- tk_local_3584x128x18944: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed

## 3584x128x3584
- cute_shape_selected_best_3584x128x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_swizzled_tn_3584x128x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_continuousc_reusea_3584x128x3584: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- tk_local_3584x128x3584: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed

## 37888x128x3584
- cute_shape_selected_best_37888x128x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_swizzled_tn_37888x128x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed

## 37888x256x3584
- cute_shape_selected_best_37888x256x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_swizzled_tn_37888x256x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_continuousc_reusea_37888x256x3584: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- tk_local_37888x256x3584: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed

## 4096x4096x4096
- cute_shape_selected_best_4096cube_bf16: cute_tk_swizzled_tn_tile128x128x128_stage4 | tflops_avg=149.486175 | err_max=0.0 | status=ok
- cute_square_tt_4096x4096x4096_bf16: cute_tk_square_tt_tile256x256x64_stage4 | tflops_avg=101.15048933333333 | err_max=2.783133 | status=ok
- cute_swizzled_tn_4096cube_bf16: cute_tk_swizzled_tn_tile128x128x128_stage4 | tflops_avg=149.130812 | err_max=0.0 | status=ok
- cute_layoutc_4096x4096x4096_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed

## 4608x128x3584
- cute_shape_selected_best_4608x128x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_swizzled_tn_4608x128x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_continuousc_reusea_4608x128x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- tk_local_continuousc_4608x128x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed

## 4608x256x3584
- cute_shape_selected_best_4608x256x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_swizzled_tn_4608x256x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_continuousc_reusea_4608x256x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- tk_local_continuousc_4608x256x3584_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed

## 8192x8192x8192
- cute_shape_selected_best_8192cube_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_swizzled_tn_8192cube_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
- cute_layoutc_8192x8192x8192_bf16: build_failed | tflops_avg=n/a | err_max=n/a | status=build_failed
