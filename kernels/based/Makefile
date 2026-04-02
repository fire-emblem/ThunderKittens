GPU ?= H100
SRC ?= linear_attn.cu
CONFIG ?= pytorch

ifeq ($(CONFIG),pytorch)
OUT ?= _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD ?= python benchmark.py
NVCCFLAGS += -DTORCH_COMPILE
else ifeq ($(CONFIG),standalone)
OUT ?= linear_attn.out
CMD ?= python gentests.py randn_all; ./linear_attn.out randn_all_1x1x1024x16x64.txt
else
$(error Unsupported CONFIG: $(CONFIG). Please set CONFIG to pytorch or standalone.)
endif

include ../common.mk
