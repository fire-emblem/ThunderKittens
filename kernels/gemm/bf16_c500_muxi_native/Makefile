GPU ?= C500
SRC := bf16_c500_muxi_native_gemm.cu
OUT := bf16_c500_muxi_native_gemm.out
CMD := ./bf16_c500_muxi_native_gemm.out
CONFIG := standalone
include ../../common.mk

ifeq ($(GPU),C500)
# Match the muxi native project build flags for BSM/global address lowering.
NVCCFLAGS += -mllvm -metaxgpu-disable-bsm-offset=0
NVCCFLAGS += -mllvm -metaxgpu-force-global-saddr=1
endif
