GPU ?= C500
SRC ?= bf16_c500_tk_local_gemm.cu
OUT ?= bf16_c500_tk_local_gemm.out
CMD ?= ./$(OUT)
CONFIG := standalone
include ../../common.mk

ifeq ($(GPU),C500)
# Match the muxi native project build flags for BSM/global address lowering.
NVCCFLAGS += -mllvm -metaxgpu-disable-bsm-offset=0
NVCCFLAGS += -mllvm -metaxgpu-force-global-saddr=1
endif
