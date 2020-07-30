# Toplevel makefile for ProjectionExplain
#
#
#
#
# Dependencies of the project on externals: CUDA and compiler.
# Normally these are the only things one should check and modify.
#
CUDAROOT  = /usr/local/cuda
CUDAINC   = $(CUDAROOT)/include
CUDAINC2  = $(CUDAROOT)/samples/common/inc
CUDALIB   = $(CUDAROOT)/lib64
CXX       = g++

#--------------------------------------------------------------------------------------------------------------------------------

# Libraries included as part of this project. They are built automatically. If however something does not work,
# enter the respective directories and build the libraries manually. See their respective makefiles.
#
ANNINC      = LIBRARIES/tsANN/include
ANNLIB    	= LIBRARIES/tsANN/lib/libANN.a
TRIANGLEINC = LIBRARIES/Triangle
TRIANGLELIB = LIBRARIES/Triangle/triangle.o
CUBUINC	   	= LIBRARIES/CUBu
CUBULIB     = LIBRARIES/CUBu/libcubu.a
GLUIINC   	= LIBRARIES/glui-master/include
GLUILIB   	= LIBRARIES/glui-master/libglui_static.a


PLATFORM  = PLATFORM_LINUX

BLDFLAGS   = -D$(PLATFORM) -m64 -O2
CMPFLAGS   = $(BLDFLAGS) -I. -Iinclude -I$(CUBUINC) -I$(ANNINC) -I$(TRIANGLEINC) -I$(CUDAINC) -I$(CUDAINC2) -I$(GLUIINC)
GLLIB      = $(GLUILIB) -lGL -lGLU -lglut -lGLEW
NVCC       = $(CUDAROOT)/bin/nvcc -ccbin $(CXX) -gencode=arch=compute_30,code=\"sm_30,compute_30\" --ptxas-options=-v --maxrregcount 50

.SUFFIXES: .o .c .cpp .cu

OBJECTS  = config.o main.o grouping.o skelft.o vis.o skelft_core.o myrandom.o image.o pointcloud.o simplepointcloud.o io.o scalarimage.o fullmatrix.o gdrawingcloud.o menu.o vgrouping.o scalar.o explanatorymap.o 

TARGET1  = projwiz
TARGET2  = concat

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(OBJECTS) $(GLUILIB) $(ANNLIB) $(TRIANGLELIB) $(CUBULIB)
	$(CXX) -o $(TARGET1) $(OBJECTS) $(BLDFLAGS) $(TRIANGLELIB) $(ANNLIB) $(CUBULIB) $(GLLIB) -L$(CUDALIB) -L$(CUDAROOT)/lib -lcudart -lstdc++

$(TARGET2): concat.o
	$(CXX) -o $(TARGET2) concat.o $(BLDFLAGS)

.cpp.o:
	$(CXX) -c $(CMPFLAGS) -Wno-deprecated -Wno-unused-result -Wno-format -o $@ $<

.cu.o:
	$(NVCC) -c $(CMPFLAGS) -o $@ $<

clean:
	-rm *.o $(TARGET1) $(TARGET2)

realclean:
	-rm *.o $(TARGET1) $(TARGET2)
	cd LIBRARIES/CUBu; make clean
	cd LIBRARIES/Triangle; make distclean
	cd LIBRARIES/tsANN; make realclean
	cd LIBRARIES/glui-master; make clean

# Rules to build the libraries which are included as part of this codebase.
# See their respective directories for their own makefiles.
#

$(GLUILIB):
	cd LIBRARIES/glui-master; make glui_static; cd -

$(ANNLIB):
	cd LIBRARIES/tsANN; make linux-g++; cd -

$(TRIANGLELIB):
	cd LIBRARIES/Triangle; make trilibrary; cd -

$(CUBULIB):
	cd LIBRARIES/CUBu; make; cd -
