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
include applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/depend.make

# Include the progress variables for this target.
include applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/progress.make

# Include the compile flags for this target's objects.
include applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/flags.make

applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o: applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/flags.make
applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o: applications/gls_navier_stokes_2d/gls_navier_stokes_2d.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o"
	cd /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o -c /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d/gls_navier_stokes_2d.cc

applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.i"
	cd /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d/gls_navier_stokes_2d.cc > CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.i

applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.s"
	cd /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d/gls_navier_stokes_2d.cc -o CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.s

applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o.requires:

.PHONY : applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o.requires

applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o.provides: applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o.requires
	$(MAKE) -f applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/build.make applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o.provides.build
.PHONY : applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o.provides

applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o.provides.build: applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o


# Object files for target gls_navier_stokes_2d
gls_navier_stokes_2d_OBJECTS = \
"CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o"

# External object files for target gls_navier_stokes_2d
gls_navier_stokes_2d_EXTERNAL_OBJECTS =

applications/gls_navier_stokes_2d/gls_navier_stokes_2d: applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/build.make
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: source/core/liblethe-core.a
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: source/solvers/liblethe-solvers.a
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: source/core/liblethe-core.a
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/dealii/build/lib/libdeal_II.so.9.2.0-pre
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/p4est/FAST/lib/libp4est.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/p4est/FAST/lib/libsc.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempif08.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempi_ignore_tkr.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_mpifh.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/libz.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/librol.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libtempus.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libmuelu-adapters.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libmuelu-interface.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libmuelu.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libmuelu_lgn.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/liblocathyra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/liblocaepetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/liblocalapack.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libloca.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libnoxepetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libnoxlapack.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libnox.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libintrepid2.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libintrepid.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libteko.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libstratimikos.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libstratimikosbelos.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libstratimikosamesos2.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libstratimikosaztecoo.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libstratimikosamesos.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libstratimikosml.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libstratimikosifpack.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libifpack2-adapters.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libifpack2.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libanasazitpetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libModeLaplace.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libanasaziepetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libanasazi.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libamesos2.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libshylu_nodetacho.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libshylu_nodehts.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libbelosxpetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libbelostpetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libbelosepetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libbelos.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libml.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libifpack.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libzoltan2.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libpamgen_extras.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libpamgen.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libamesos.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libgaleri-xpetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libgaleri-epetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libaztecoo.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libisorropia.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libxpetra-sup.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libxpetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libthyratpetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libthyraepetraext.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libthyraepetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libthyracore.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libtrilinosss.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libtpetraext.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libtpetrainout.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libtpetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libkokkostsqr.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libtpetraclassiclinalg.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libtpetraclassicnodeapi.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libtpetraclassic.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libepetraext.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libtriutils.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libshards.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libzoltan.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libepetra.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libminitensor.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libsacado.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/librtop.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libkokkoskernels.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libteuchoskokkoscomm.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libteuchoskokkoscompat.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libteuchosremainder.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libteuchosnumerics.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libteuchoscomm.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libteuchosparameterlist.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libteuchosparser.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libteuchoscore.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libkokkosalgorithms.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libkokkoscontainers.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libkokkoscore.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /home/shahab/share/trilinos/lib/libgtest.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/libarpack.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/libnetcdf.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/liblapack.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/libblas.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
applications/gls_navier_stokes_2d/gls_navier_stokes_2d: applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable gls_navier_stokes_2d"
	cd /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gls_navier_stokes_2d.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/build: applications/gls_navier_stokes_2d/gls_navier_stokes_2d

.PHONY : applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/build

applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/requires: applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/gls_navier_stokes_2d.cc.o.requires

.PHONY : applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/requires

applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/clean:
	cd /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d && $(CMAKE_COMMAND) -P CMakeFiles/gls_navier_stokes_2d.dir/cmake_clean.cmake
.PHONY : applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/clean

applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/depend:
	cd /home/shahab/Lethe_newversion/lethe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : applications/gls_navier_stokes_2d/CMakeFiles/gls_navier_stokes_2d.dir/depend

