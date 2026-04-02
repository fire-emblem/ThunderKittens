GPU ?= H100
SRC ?= fftconv_pc.cu
CONFIG ?= pytorch

ifeq ($(CONFIG),pytorch)
OUT ?= _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD ?= python benchmark.py
NVCCFLAGS += -DTORCH_COMPILE
else ifeq ($(CONFIG),standalone)
OUT ?= fftconv_pc.out
CMD ?= python gentests.py randn; ./fftconv_pc.out randn_f4096_H1024_B4.txt
else
$(error Unsupported CONFIG: $(CONFIG). Please set CONFIG to pytorch or standalone.)
endif

include ../common.mk
