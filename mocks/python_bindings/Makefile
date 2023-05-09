ROOT_DIR := ../..
UTILS_DIR := $(ROOT_DIR)/utils
IO_DIR := $(ROOT_DIR)/io
LIB_DIR := $(ROOT_DIR)/lib
DATA_DIR := ../tests/data
INSTALL_HEADERS_DIR := $(ROOT_DIR)/include
INSTALL_LIB_DIR := $(ROOT_DIR)/lib
INSTALL_BIN_DIR := $(ROOT_DIR)/bin

MOCKS_DIR := $(ROOT_DIR)/mocks
DDleg_mocks_DIR := $(MOCKS_DIR)/DDleg_mocks
DDbessel_mocks_DIR := $(MOCKS_DIR)/DDbessel_mocks
DDrppi_mocks_DIR := $(MOCKS_DIR)/DDrppi_mocks
DDtheta_mocks_DIR := $(MOCKS_DIR)/DDtheta_mocks
DDsmu_mocks_DIR := $(MOCKS_DIR)/DDsmu_mocks
VPF_mocks_DIR := $(MOCKS_DIR)/vpf_mocks

DDleg_mocks_LIB := countpairs_leg_mocks
DDbessel_mocks_LIB := countpairs_bessel_mocks
DDrppi_mocks_LIB := countpairs_rp_pi_mocks
DDsmu_mocks_LIB := countpairs_s_mu_mocks
DDtheta_mocks_LIB := countpairs_theta_mocks
VPF_mocks_LIB := countspheres_mocks

GSL_REQUIRED := true
include $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk

PROJECT := _countpairs_mocks
PYTHON_EXTN := $(PROJECT).so.$(MAJOR).$(MINOR).$(PATCHLEVEL)
SOURCES := $(PROJECT).c
OBJECTS := $(SOURCES:.c=.o)
C_LIBRARIES := $(DDleg_mocks_DIR)/lib$(DDleg_mocks_LIB).a $(DDbessel_mocks_DIR)/lib$(DDbessel_mocks_LIB).a $(DDrppi_mocks_DIR)/lib$(DDrppi_mocks_LIB).a $(DDtheta_mocks_DIR)/lib$(DDtheta_mocks_LIB).a $(VPF_mocks_DIR)/lib$(VPF_mocks_LIB).a $(DDsmu_mocks_DIR)/lib$(DDsmu_mocks_LIB).a
INCL :=	$(DDleg_mocks_DIR)/$(DDleg_mocks_LIB).h $(DDbessel_mocks_DIR)/$(DDbessel_mocks_LIB).h $(DDsmu_mocks_DIR)/$(DDsmu_mocks_LIB).h $(DDrppi_mocks_DIR)/$(DDrppi_mocks_LIB).h $(DDtheta_mocks_DIR)/$(DDtheta_mocks_LIB).h $(VPF_mocks_DIR)/$(VPF_mocks_LIB).h \
        $(UTILS_DIR)/defs.h $(IO_DIR)/io.h $(IO_DIR)/ftread.h \
        $(UTILS_DIR)/utils.h \
		$(UTILS_DIR)/function_precision.h \
        $(UTILS_DIR)/progressbar.h \
        $(UTILS_DIR)/cpu_features.h $(UTILS_DIR)/macros.h \

LIB_INCLUDE:=-I$(DDleg_mocks_DIR) -I$(DDbessel_mocks_DIR) -I$(DDrppi_mocks_DIR) -I$(DDtheta_mocks_DIR) -I$(VPF_mocks_DIR) -I$(DDsmu_mocks_DIR)

all: sharedlib $(LIBRARY) $(SOURCES) $(C_LIBRARIES) $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk Makefile

$(PROJECT).so: $(PYTHON_EXTN)
	$(RM) $(PROJECT).so
	ln -s $(PYTHON_EXTN) $(PROJECT).so

$(PROJECT).o: $(PROJECT).c $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk Makefile $(C_LIBRARIES) $(INCL)
	$(CC) $(CFLAGS) $(LIB_INCLUDE) $(PYTHON_CFLAGS)  $(INCLUDE) $< -c -o $@

tests: sharedlib $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk Makefile $(C_LIBRARIES) $(OBJECTS) | $(ROOT_DIR)/lib

$(PYTHON_EXTN): $(OBJECTS) $(C_LIBRARIES) $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk Makefile
	$(CC) $(OBJECTS) $(C_LIBRARIES) $(CLINK) $(GSL_LINK) $(PYTHON_LINK) -shared -o $@

$(DDleg_mocks_DIR)/lib$(DDleg_mocks_LIB).a: $(DDleg_mocks_DIR)/*.c $(DDleg_mocks_DIR)/*.c.src $(DDleg_mocks_DIR)/*.h.src $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk
	$(MAKE) -C $(DDleg_mocks_DIR) libs

$(DDbessel_mocks_DIR)/lib$(DDbessel_mocks_LIB).a: $(DDbessel_mocks_DIR)/*.c $(DDbessel_mocks_DIR)/*.c.src $(DDbessel_mocks_DIR)/*.h.src $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk
	$(MAKE) -C $(DDbessel_mocks_DIR) libs

$(DDrppi_mocks_DIR)/lib$(DDrppi_mocks_LIB).a: $(DDrppi_mocks_DIR)/*.c $(DDrppi_mocks_DIR)/*.c.src $(DDrppi_mocks_DIR)/*.h.src $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk
	$(MAKE) -C $(DDrppi_mocks_DIR) libs

$(DDsmu_mocks_DIR)/lib$(DDsmu_mocks_LIB).a: $(DDsmu_mocks_DIR)/*.c $(DDsmu_mocks_DIR)/*.c.src $(DDsmu_mocks_DIR)/*.h.src $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk
	$(MAKE) -C $(DDsmu_mocks_DIR) libs

$(DDtheta_mocks_DIR)/lib$(DDtheta_mocks_LIB).a: $(DDtheta_mocks_DIR)/*.c $(DDtheta_mocks_DIR)/*.c.src $(DDtheta_mocks_DIR)/*.h.src $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk
	$(MAKE) -C $(DDtheta_mocks_DIR) libs

$(VPF_mocks_DIR)/lib$(VPF_mocks_LIB).a: $(VPF_mocks_DIR)/*.c $(VPF_mocks_DIR)/*.c.src $(VPF_mocks_DIR)/*.h.src $(ROOT_DIR)/mocks.options $(ROOT_DIR)/common.mk
	$(MAKE) -C $(VPF_mocks_DIR) libs

install: sharedlib $(LIB_DIR)/$(PYTHON_EXTN)
sharedlib: $(PROJECT).so

.PHONY: sharedlib

$(INSTALL_LIB_DIR)/$(PYTHON_EXTN): $(PYTHON_EXTN) | $(INSTALL_LIB_DIR)
	cp -p $(PYTHON_EXTN) $(INSTALL_LIB_DIR)/

$(INSTALL_LIB_DIR)/$(PROJECT).so:$(INSTALL_LIB_DIR)/$(PYTHON_EXTN)
	$(RM) $(INSTALL_LIB_DIR)/$(PROJECT).so
	cd $(INSTALL_LIB_DIR) && ln -s $(PYTHON_EXTN) $(PROJECT).so


distclean: clean | $(LIB_DIR)
	cd $(INSTALL_LIB_DIR) && $(RM) $(PYTHON_EXTN) $(PROJECT).so
	cd ../../Corrfunc && $(RM) $(PROJECT).so

clean:
	$(RM) $(PYTHON_EXTN) $(OBJECTS) $(PROJECT).so
	$(RM) -R *.dSYM

include $(ROOT_DIR)/rules.mk
