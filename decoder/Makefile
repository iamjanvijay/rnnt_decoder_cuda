DEBUG:= -Xcompiler="-fpermissive"

TARGET    = decoder

SRC_DIR_MAIN   = ./
SRC_DIR_SYS   = ../sys
SRC_DIR_COMMON   = ../common

OBJ_DIR_MAIN   = ./obj
OBJ_DIR_SYS   = ../sys/obj
OBJ_DIR_COMMON   = ../common/obj

INCLUDES:=-I../common/header -I../sys/header -I../ext/include/ -I./header/

NVCC:=nvcc
LDFLAGS:= -lcudnn -lcublas -lcurand 
NVCCFLAGS:= -arch=sm_70 --use_fast_math -std=c++14 -O2
CUDNN_PATH:= /usr/local/cuda/
LIBS:= -L $(CUDNN_PATH)/lib64 -L/usr/local/lib -L ../ext/lib/

EXTRA_FLAGS:= $(DEBUG)
	

CU_FILES_MAIN_H  = $(wildcard $(SRC_DIR_MAIN)/header/*.hpp)
CU_FILES_MAIN  = $(wildcard $(SRC_DIR_MAIN)/src/*.cu)
OBJS_MAIN = $(patsubst %.cu,$(OBJ_DIR_MAIN)/%.o,$(notdir $(CU_FILES_MAIN)))

CU_FILES_SYS_H  = $(wildcard $(SRC_DIR_SYS)/header/*.hpp)
CU_FILES_SYS  = $(wildcard $(SRC_DIR_SYS)/src/*.cu)
OBJS_SYS = $(patsubst %.cu,$(OBJ_DIR_SYS)/%.o,$(notdir $(CU_FILES_SYS)))

CU_FILES_COMMON_H  = $(wildcard $(SRC_DIR_COMMON)/header/*.hpp)
CU_FILES_COMMON  = $(wildcard $(SRC_DIR_COMMON)/src/*.cpp)
OBJS_COMMON = $(patsubst %.cpp,$(OBJ_DIR_COMMON)/%.o,$(notdir $(CU_FILES_COMMON)))

$(TARGET) : EXTRA_FLAGS += -D T2_CN
$(TARGET) :	dirmake cleanmain $(OBJS_COMMON)	$(OBJS_SYS) $(OBJS_MAIN)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(LIBS) -o $@ $(OBJS_COMMON) $(OBJS_SYS) $(OBJS_MAIN)

$(OBJ_DIR_SYS)/%.o : $(SRC_DIR_SYS)/src/%.cu $(CU_FILES_SYS_H) $(CU_FILES_COMMON_H)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $< $(EXTRA_FLAGS)

$(OBJ_DIR_COMMON)/%.o : $(SRC_DIR_COMMON)/src/%.cpp $(CU_FILES_COMMON_H)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $< $(EXTRA_FLAGS)

$(OBJ_DIR_MAIN)/%.o : $(SRC_DIR_MAIN)/src/%.cu $(CU_FILES_MAIN_H)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I./ -o $@ -c $< $(EXTRA_FLAGS)


dirmake:
	@mkdir -p $(OBJ_DIR_COMMON)
	@mkdir -p $(OBJ_DIR_SYS)
	@mkdir -p $(OBJ_DIR_MAIN)

cleanmain:
	rm -f $(OBJ_DIR_MAIN)/*.o


.PHONY :	clean
clean :	
	rm -f $(TARGET)
	rm -f $(OBJ_DIR_COMMON)/*.o
	rm -f $(OBJ_DIR_SYS)/*.o
	rm -f $(OBJ_DIR_MAIN)/*.o

rebuild:	clean build
