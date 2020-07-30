
#--------------------------------------------------------------------------------------------------------------------------------

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
PLATFORM = PLATFORM_LINUX
endif
ifeq ($(UNAME), Darwin)
PLATFORM = PLATFORM_OSX
endif

ifeq ($(PLATFORM),PLATFORM_LINUX)
CXX       =  g++
CUDAROOT  =  /usr/local/cuda
CUDAINC   = -I$(CUDAROOT)/samples/common/inc
CUDALIB   = $(CUDAROOT)/lib64
GLUI      = /usr/local/glui
GLINC     = -I$(GLUI)/include
GLLIB     = $(GLUI)/lib/libglui.a -lGL -lGLU -lglut -lGLEW
CUDAGEN   = -gencode=arch=compute_30,code=\"sm_30,compute_30\"
endif

ifeq ($(PLATFORM),PLATFORM_OSX)

# ALEX: Apparently on my Mac it's critical to have
#       export MACOSX_DEPLOYMENT_TARGET=10.7
#       done before building, otherwise clang complains about tr1/ headers and the like.

CXX       = /usr/bin/clang++
CUDAROOT  = /Developer/NVIDIA/CUDA-9.0
CUDAINC   = -I$(CUDAROOT)/samples/common/inc 
CUDALIB   = $(CUDAROOT)/lib64
GLINC     = 
GLLIB     = -framework OpenGL -framework GLUT -framework GLUI /usr/lib/libGLEW.a
CUDAGEN   = #-gencode arch=compute_12,code=\"sm_12,compute_12\"
endif


#--------------------------------------------------------------------------------------------------------------------------------

CCFLAGS   = -I. -Iinclude -m64 -O2 -I$(CUDAROOT)/include $(CUDAINC) -D$(PLATFORM) 
CXXFLAGS  = $(CCFLAGS) $(GLINC)
NVCC = $(CUDAROOT)/bin/nvcc -ccbin $(CXX)
.SUFFIXES: .o .c .cpp .cu


LIBOBJECTS  = gdrawing.o cpubundling.o gpubundling.o sparsematrix.o polyline.o utils.o
DEMOOBJECTS = main.o

DEMO	= cubu
CUBULIB = libcubu.a




all: $(DEMO) $(CUBULIB)

$(CUBULIB): $(LIBOBJECTS)
	ar -rv $(CUBULIB) $(LIBOBJECTS)

$(DEMO): $(CUBULIB) $(DEMOOBJECTS)
	$(CXX) -o $(DEMO) $(DEMOOBJECTS) $(CUBULIB) $(CXXFLAGS) -L$(CUDALIB) $(GLLIB) -L$(CUDAROOT)/lib -lcudart -lstdc++

.cpp.o:
	$(CXX) -c $(CXXFLAGS)  -o $@ $<

.cu.o:
	$(NVCC) -c $(CCFLAGS) -use_fast_math $(CUDAGEN) --ptxas-options=-v --maxrregcount 50 -o $@ $<

clean:
	-@rm *.o $(DEMO) $(CUBULIB) 2>/dev/null || true

