GPU ?= H100
SRC ?= flux_gate.cu
CONFIG ?= pytorch

ifeq ($(CONFIG),pytorch)
OUT ?= _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD ?= python benchmark.py
NVCCFLAGS += -DTORCH_COMPILE
else ifeq ($(CONFIG),standalone)
OUT ?= flux_gate.out
CMD ?= ./flux_gate.out
NVCCFLAGS += -fopenmp -lgomp
else
$(error Unsupported CONFIG: $(CONFIG). Please set CONFIG to pytorch or standalone.)
endif

include ../common.mk
