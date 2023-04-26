ROOT_DIR := ../..
UTILS_DIR := $(ROOT_DIR)/utils
IO_DIR := $(ROOT_DIR)/io
INCLUDE_DIR := $(ROOT_DIR)/include
LIB_DIR := $(ROOT_DIR)/lib
DATA_DIR := ../tests/data

MOCKS_DIR := $(ROOT_DIR)/mocks
DDrppi_mocks_DIR := $(MOCKS_DIR)/DDrppi_mocks
DDsmu_mocks_DIR := $(MOCKS_DIR)/DDsmu_mocks
DDTHETA_mocks_DIR := $(MOCKS_DIR)/DDtheta_mocks
VPF_mocks_DIR := $(MOCKS_DIR)/vpf_mocks

DDrppi_mocks_LIB := countpairs_rp_pi_mocks
DDsmu_mocks_LIB := countpairs_s_mu_mocks
DDTHETA_mocks_LIB := countpairs_theta_mocks
VPF_mocks_LIB := countspheres_mocks

GSL_REQUIRED := false
include $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk

TARGET := run_correlations_mocks
TARGETSRC := run_correlations_mocks.c $(IO_DIR)/ftread.c $(IO_DIR)/io.c $(UTILS_DIR)/utils.c $(UTILS_DIR)/progressbar.c
TARGETOBJS  := $(TARGETSRC:.c=.o)
TARGET_THETA := run_correlations_theta_mocks
TARGETSRC_THETA := run_correlations_theta_mocks.c $(IO_DIR)/ftread.c $(IO_DIR)/io.c $(UTILS_DIR)/utils.c $(UTILS_DIR)/progressbar.c
TARGETOBJS_THETA := $(TARGETSRC_THETA:.c=.o)
C_LIBRARIES := $(DDrppi_mocks_DIR)/lib$(DDrppi_mocks_LIB).a $(DDsmu_mocks_DIR)/lib$(DDsmu_mocks_LIB).a \
               $(DDTHETA_mocks_DIR)/lib$(DDTHETA_mocks_LIB).a $(VPF_mocks_DIR)/lib$(VPF_mocks_LIB).a
INCL :=	$(DDrppi_mocks_DIR)/$(DDrppi_mocks_LIB).h $(DDsmu_mocks_DIR)/$(DDsmu_mocks_LIB).h \
        $(DDTHETA_mocks_DIR)/$(DDTHETA_mocks_LIB).h $(VPF_mocks_DIR)/$(VPF_mocks_LIB).h \
        $(UTILS_DIR)/defs.h $(IO_DIR)/io.h $(IO_DIR)/ftread.h \
        $(UTILS_DIR)/utils.h $(UTILS_DIR)/gridlink_mocks_impl_double.h $(UTILS_DIR)/gridlink_mocks_impl_float.h \
		$(UTILS_DIR)/function_precision.h $(UTILS_DIR)/cellarray_mocks_double.h $(UTILS_DIR)/cellarray_mocks_float.h \
        $(UTILS_DIR)/progressbar.h

LIBRARY_INCL := -I$(DDrppi_mocks_DIR) -I$(DDsmu_mocks_DIR) -I$(DDTHETA_mocks_DIR) -I$(VPF_mocks_DIR)

all: $(TARGET) $(TARGETSRC) $(TARGET_THETA) $(TARGETSRC_THETA) $(C_LIBRARIES) $(INCL) $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk Makefile

EXTRA_LINK:= $(GSL_LINK)
EXTRA_INCL:= $(GSL_CFLAGS) $(LIBRARY_INCL)

$(TARGET): $(TARGETOBJS) $(ROOT_DIR)/common.mk
	$(CC) $(TARGETOBJS) $(C_LIBRARIES) $(CLINK) $(EXTRA_LINK) -o $@

$(TARGET).o: $(TARGET).c $(ROOT_DIR)/common.mk Makefile $(ROOT_DIR)/theory.options $(ROOT_DIR)/mocks.options
	$(CC) $(OPT) $(CFLAGS) $(INCLUDE) $(EXTRA_INCL) -c $< -o $@

$(TARGET_THETA): $(TARGETOBJS_THETA) $(ROOT_DIR)/common.mk
	$(CC) $(TARGETOBJS_THETA) $(C_LIBRARIES) $(CLINK) $(EXTRA_LINK) -o $@

$(TARGET_THETA).o: $(TARGET_THETA).c $(ROOT_DIR)/common.mk Makefile $(ROOT_DIR)/theory.options $(ROOT_DIR)/mocks.options
	$(CC) $(OPT) $(CFLAGS) $(INCLUDE) $(EXTRA_INCL) -c $< -o $@

$(DDrppi_mocks_DIR)/lib$(DDrppi_mocks_LIB).a: $(DDrppi_mocks_DIR)/*.c $(DDrppi_mocks_DIR)/*.c.src $(DDrppi_mocks_DIR)/*.h.src $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk
	$(MAKE) -C $(DDrppi_mocks_DIR) libs

$(DDsmu_mocks_DIR)/lib$(DDsmu_mocks_LIB).a: $(DDsmu_mocks_DIR)/*.c $(DDsmu_mocks_DIR)/*.c.src $(DDsmu_mocks_DIR)/*.h.src $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk
	$(MAKE) -C $(DDsmu_mocks_DIR) libs

$(DDTHETA_mocks_DIR)/lib$(DDTHETA_mocks_LIB).a: $(DDTHETA_mocks_DIR)/*.c $(DDTHETA_mocks_DIR)/*.c.src $(DDTHETA_mocks_DIR)/*.h.src $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk
	$(MAKE) -C $(DDTHETA_mocks_DIR) libs

$(VPF_mocks_DIR)/lib$(VPF_mocks_LIB).a: $(VPF_mocks_DIR)/*.c $(VPF_mocks_DIR)/*.c.src $(VPF_mocks_DIR)/*.h.src $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk
	$(MAKE) -C $(VPF_mocks_DIR) libs

clean:
	$(RM) $(TARGET) $(TARGETOBJS) $(TARGET_THETA) $(TARGETOBJS_THETA)
	$(RM) -R *.dSYM

distclean:clean

include $(ROOT_DIR)/rules.mk
