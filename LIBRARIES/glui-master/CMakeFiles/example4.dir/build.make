# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master

# Include any dependencies generated for this target.
include CMakeFiles/example4.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/example4.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/example4.dir/flags.make

CMakeFiles/example4.dir/example/example4.cpp.o: CMakeFiles/example4.dir/flags.make
CMakeFiles/example4.dir/example/example4.cpp.o: example/example4.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/example4.dir/example/example4.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example4.dir/example/example4.cpp.o -c /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master/example/example4.cpp

CMakeFiles/example4.dir/example/example4.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example4.dir/example/example4.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master/example/example4.cpp > CMakeFiles/example4.dir/example/example4.cpp.i

CMakeFiles/example4.dir/example/example4.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example4.dir/example/example4.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master/example/example4.cpp -o CMakeFiles/example4.dir/example/example4.cpp.s

CMakeFiles/example4.dir/example/example4.cpp.o.requires:

.PHONY : CMakeFiles/example4.dir/example/example4.cpp.o.requires

CMakeFiles/example4.dir/example/example4.cpp.o.provides: CMakeFiles/example4.dir/example/example4.cpp.o.requires
	$(MAKE) -f CMakeFiles/example4.dir/build.make CMakeFiles/example4.dir/example/example4.cpp.o.provides.build
.PHONY : CMakeFiles/example4.dir/example/example4.cpp.o.provides

CMakeFiles/example4.dir/example/example4.cpp.o.provides.build: CMakeFiles/example4.dir/example/example4.cpp.o


# Object files for target example4
example4_OBJECTS = \
"CMakeFiles/example4.dir/example/example4.cpp.o"

# External object files for target example4
example4_EXTERNAL_OBJECTS =

example4: CMakeFiles/example4.dir/example/example4.cpp.o
example4: CMakeFiles/example4.dir/build.make
example4: libglui_static.a
example4: /usr/lib/x86_64-linux-gnu/libglut.so
example4: /usr/lib/x86_64-linux-gnu/libGL.so
example4: /usr/lib/x86_64-linux-gnu/libGLU.so
example4: CMakeFiles/example4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example4"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/example4.dir/build: example4

.PHONY : CMakeFiles/example4.dir/build

CMakeFiles/example4.dir/requires: CMakeFiles/example4.dir/example/example4.cpp.o.requires

.PHONY : CMakeFiles/example4.dir/requires

CMakeFiles/example4.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/example4.dir/cmake_clean.cmake
.PHONY : CMakeFiles/example4.dir/clean

CMakeFiles/example4.dir/depend:
	cd /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master /home/tzloop/Desktop/FNL-ProjectionExplain/LIBRARIES/glui-master/CMakeFiles/example4.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/example4.dir/depend
