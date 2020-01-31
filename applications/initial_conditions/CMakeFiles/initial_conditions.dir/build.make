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
include applications/initial_conditions/CMakeFiles/initial_conditions.dir/depend.make

# Include the progress variables for this target.
include applications/initial_conditions/CMakeFiles/initial_conditions.dir/progress.make

# Include the compile flags for this target's objects.
include applications/initial_conditions/CMakeFiles/initial_conditions.dir/flags.make

applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o: applications/initial_conditions/CMakeFiles/initial_conditions.dir/flags.make
applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o: applications/initial_conditions/initial_conditions.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o"
	cd /home/shahab/Lethe_newversion/lethe/applications/initial_conditions && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/initial_conditions.dir/initial_conditions.cc.o -c /home/shahab/Lethe_newversion/lethe/applications/initial_conditions/initial_conditions.cc

applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/initial_conditions.dir/initial_conditions.cc.i"
	cd /home/shahab/Lethe_newversion/lethe/applications/initial_conditions && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shahab/Lethe_newversion/lethe/applications/initial_conditions/initial_conditions.cc > CMakeFiles/initial_conditions.dir/initial_conditions.cc.i

applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/initial_conditions.dir/initial_conditions.cc.s"
	cd /home/shahab/Lethe_newversion/lethe/applications/initial_conditions && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shahab/Lethe_newversion/lethe/applications/initial_conditions/initial_conditions.cc -o CMakeFiles/initial_conditions.dir/initial_conditions.cc.s

applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o.requires:

.PHONY : applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o.requires

applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o.provides: applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o.requires
	$(MAKE) -f applications/initial_conditions/CMakeFiles/initial_conditions.dir/build.make applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o.provides.build
.PHONY : applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o.provides

applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o.provides.build: applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o


# Object files for target initial_conditions
initial_conditions_OBJECTS = \
"CMakeFiles/initial_conditions.dir/initial_conditions.cc.o"

# External object files for target initial_conditions
initial_conditions_EXTERNAL_OBJECTS =

applications/initial_conditions/initial_conditions: applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o
applications/initial_conditions/initial_conditions: applications/initial_conditions/CMakeFiles/initial_conditions.dir/build.make
applications/initial_conditions/initial_conditions: source/core/liblethe-core.a
applications/initial_conditions/initial_conditions: source/solvers/liblethe-solvers.a
applications/initial_conditions/initial_conditions: source/core/liblethe-core.a
applications/initial_conditions/initial_conditions: /home/shahab/dealii/build/lib/libdeal_II.so.9.2.0-pre
applications/initial_conditions/initial_conditions: /home/shahab/p4est/FAST/lib/libp4est.so
applications/initial_conditions/initial_conditions: /home/shahab/p4est/FAST/lib/libsc.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempif08.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempi_ignore_tkr.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_mpifh.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/libz.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/librol.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libtempus.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libmuelu-adapters.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libmuelu-interface.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libmuelu.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libmuelu_lgn.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/liblocathyra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/liblocaepetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/liblocalapack.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libloca.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libnoxepetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libnoxlapack.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libnox.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libintrepid2.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libintrepid.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libteko.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libstratimikos.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libstratimikosbelos.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libstratimikosamesos2.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libstratimikosaztecoo.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libstratimikosamesos.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libstratimikosml.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libstratimikosifpack.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libifpack2-adapters.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libifpack2.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libanasazitpetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libModeLaplace.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libanasaziepetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libanasazi.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libamesos2.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libshylu_nodetacho.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libshylu_nodehts.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libbelosxpetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libbelostpetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libbelosepetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libbelos.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libml.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libifpack.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libzoltan2.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libpamgen_extras.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libpamgen.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libamesos.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libgaleri-xpetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libgaleri-epetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libaztecoo.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libisorropia.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libxpetra-sup.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libxpetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libthyratpetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libthyraepetraext.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libthyraepetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libthyracore.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libtrilinosss.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libtpetraext.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libtpetrainout.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libtpetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libkokkostsqr.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libtpetraclassiclinalg.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libtpetraclassicnodeapi.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libtpetraclassic.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libepetraext.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libtriutils.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libshards.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libzoltan.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libepetra.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libminitensor.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libsacado.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/librtop.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libkokkoskernels.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libteuchoskokkoscomm.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libteuchoskokkoscompat.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libteuchosremainder.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libteuchosnumerics.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libteuchoscomm.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libteuchosparameterlist.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libteuchosparser.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libteuchoscore.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libkokkosalgorithms.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libkokkoscontainers.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libkokkoscore.so
applications/initial_conditions/initial_conditions: /home/shahab/share/trilinos/lib/libgtest.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/libarpack.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/libnetcdf.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/liblapack.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/libblas.so
applications/initial_conditions/initial_conditions: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
applications/initial_conditions/initial_conditions: applications/initial_conditions/CMakeFiles/initial_conditions.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable initial_conditions"
	cd /home/shahab/Lethe_newversion/lethe/applications/initial_conditions && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/initial_conditions.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
applications/initial_conditions/CMakeFiles/initial_conditions.dir/build: applications/initial_conditions/initial_conditions

.PHONY : applications/initial_conditions/CMakeFiles/initial_conditions.dir/build

applications/initial_conditions/CMakeFiles/initial_conditions.dir/requires: applications/initial_conditions/CMakeFiles/initial_conditions.dir/initial_conditions.cc.o.requires

.PHONY : applications/initial_conditions/CMakeFiles/initial_conditions.dir/requires

applications/initial_conditions/CMakeFiles/initial_conditions.dir/clean:
	cd /home/shahab/Lethe_newversion/lethe/applications/initial_conditions && $(CMAKE_COMMAND) -P CMakeFiles/initial_conditions.dir/cmake_clean.cmake
.PHONY : applications/initial_conditions/CMakeFiles/initial_conditions.dir/clean

applications/initial_conditions/CMakeFiles/initial_conditions.dir/depend:
	cd /home/shahab/Lethe_newversion/lethe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/applications/initial_conditions /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/applications/initial_conditions /home/shahab/Lethe_newversion/lethe/applications/initial_conditions/CMakeFiles/initial_conditions.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : applications/initial_conditions/CMakeFiles/initial_conditions.dir/depend

