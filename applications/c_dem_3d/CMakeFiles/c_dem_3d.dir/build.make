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
CMAKE_SOURCE_DIR = /home/shahab/Lethe_newversion/lethe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shahab/Lethe_newversion/lethe

# Include any dependencies generated for this target.
include applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/depend.make

# Include the progress variables for this target.
include applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/progress.make

# Include the compile flags for this target's objects.
include applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/flags.make

applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o: applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/flags.make
applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o: applications/c_dem_3d/dem_3d.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o"
	cd /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/c_dem_3d.dir/dem_3d.cc.o -c /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d/dem_3d.cc

applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/c_dem_3d.dir/dem_3d.cc.i"
	cd /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d/dem_3d.cc > CMakeFiles/c_dem_3d.dir/dem_3d.cc.i

applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/c_dem_3d.dir/dem_3d.cc.s"
	cd /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d/dem_3d.cc -o CMakeFiles/c_dem_3d.dir/dem_3d.cc.s

applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o.requires:

.PHONY : applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o.requires

applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o.provides: applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o.requires
	$(MAKE) -f applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/build.make applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o.provides.build
.PHONY : applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o.provides

applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o.provides.build: applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o


# Object files for target c_dem_3d
c_dem_3d_OBJECTS = \
"CMakeFiles/c_dem_3d.dir/dem_3d.cc.o"

# External object files for target c_dem_3d
c_dem_3d_EXTERNAL_OBJECTS =

applications/c_dem_3d/c_dem_3d: applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o
applications/c_dem_3d/c_dem_3d: applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/build.make
applications/c_dem_3d/c_dem_3d: source/core/liblethe-core.a
applications/c_dem_3d/c_dem_3d: source/dem/liblethe-dem.a
applications/c_dem_3d/c_dem_3d: /home/shahab/dealii/build/lib/libdeal_II.so.9.2.0-pre
applications/c_dem_3d/c_dem_3d: /home/shahab/p4est/FAST/lib/libp4est.so
applications/c_dem_3d/c_dem_3d: /home/shahab/p4est/FAST/lib/libsc.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempif08.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempi_ignore_tkr.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_mpifh.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/libz.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/librol.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libtempus.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libmuelu-adapters.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libmuelu-interface.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libmuelu.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libmuelu_lgn.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/liblocathyra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/liblocaepetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/liblocalapack.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libloca.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libnoxepetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libnoxlapack.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libnox.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libintrepid2.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libintrepid.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libteko.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libstratimikos.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libstratimikosbelos.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libstratimikosamesos2.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libstratimikosaztecoo.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libstratimikosamesos.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libstratimikosml.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libstratimikosifpack.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libifpack2-adapters.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libifpack2.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libanasazitpetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libModeLaplace.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libanasaziepetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libanasazi.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libamesos2.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libshylu_nodetacho.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libshylu_nodehts.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libbelosxpetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libbelostpetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libbelosepetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libbelos.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libml.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libifpack.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libzoltan2.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libpamgen_extras.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libpamgen.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libamesos.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libgaleri-xpetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libgaleri-epetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libaztecoo.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libisorropia.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libxpetra-sup.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libxpetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libthyratpetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libthyraepetraext.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libthyraepetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libthyracore.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libtrilinosss.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libtpetraext.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libtpetrainout.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libtpetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libkokkostsqr.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libtpetraclassiclinalg.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libtpetraclassicnodeapi.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libtpetraclassic.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libepetraext.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libtriutils.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libshards.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libzoltan.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libepetra.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libminitensor.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libsacado.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/librtop.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libkokkoskernels.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libteuchoskokkoscomm.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libteuchoskokkoscompat.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libteuchosremainder.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libteuchosnumerics.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libteuchoscomm.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libteuchosparameterlist.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libteuchosparser.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libteuchoscore.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libkokkosalgorithms.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libkokkoscontainers.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libkokkoscore.so
applications/c_dem_3d/c_dem_3d: /home/shahab/share/trilinos/lib/libgtest.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/libarpack.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/libnetcdf.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/liblapack.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/libblas.so
applications/c_dem_3d/c_dem_3d: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
applications/c_dem_3d/c_dem_3d: applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable c_dem_3d"
	cd /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c_dem_3d.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/build: applications/c_dem_3d/c_dem_3d

.PHONY : applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/build

applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/requires: applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/dem_3d.cc.o.requires

.PHONY : applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/requires

applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/clean:
	cd /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d && $(CMAKE_COMMAND) -P CMakeFiles/c_dem_3d.dir/cmake_clean.cmake
.PHONY : applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/clean

applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/depend:
	cd /home/shahab/Lethe_newversion/lethe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d /home/shahab/Lethe_newversion/lethe/applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : applications/c_dem_3d/CMakeFiles/c_dem_3d.dir/depend

