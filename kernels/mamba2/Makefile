GPU ?= H100
SRC ?= mamba2.cu
CONFIG ?= pytorch

ifeq ($(CONFIG),pytorch)
OUT ?= _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD ?= python benchmark.py
NVCCFLAGS += -DTORCH_COMPILE
else ifeq ($(CONFIG),standalone)
OUT ?= mamba2.out
CMD ?= python gentests.py; ./mamba2.out randn_1024N_64D_64S.txt
else
$(error Unsupported CONFIG: $(CONFIG). Please set CONFIG to pytorch or standalone.)
endif

include ../common.mk
