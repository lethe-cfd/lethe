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
include prototypes/template/CMakeFiles/protype_template.dir/depend.make

# Include the progress variables for this target.
include prototypes/template/CMakeFiles/protype_template.dir/progress.make

# Include the compile flags for this target's objects.
include prototypes/template/CMakeFiles/protype_template.dir/flags.make

prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o: prototypes/template/CMakeFiles/protype_template.dir/flags.make
prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o: prototypes/template/prototype_template.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o"
	cd /home/shahab/Lethe_newversion/lethe/prototypes/template && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/protype_template.dir/prototype_template.cc.o -c /home/shahab/Lethe_newversion/lethe/prototypes/template/prototype_template.cc

prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/protype_template.dir/prototype_template.cc.i"
	cd /home/shahab/Lethe_newversion/lethe/prototypes/template && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shahab/Lethe_newversion/lethe/prototypes/template/prototype_template.cc > CMakeFiles/protype_template.dir/prototype_template.cc.i

prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/protype_template.dir/prototype_template.cc.s"
	cd /home/shahab/Lethe_newversion/lethe/prototypes/template && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shahab/Lethe_newversion/lethe/prototypes/template/prototype_template.cc -o CMakeFiles/protype_template.dir/prototype_template.cc.s

prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o.requires:

.PHONY : prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o.requires

prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o.provides: prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o.requires
	$(MAKE) -f prototypes/template/CMakeFiles/protype_template.dir/build.make prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o.provides.build
.PHONY : prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o.provides

prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o.provides.build: prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o


# Object files for target protype_template
protype_template_OBJECTS = \
"CMakeFiles/protype_template.dir/prototype_template.cc.o"

# External object files for target protype_template
protype_template_EXTERNAL_OBJECTS =

prototypes/template/protype_template: prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o
prototypes/template/protype_template: prototypes/template/CMakeFiles/protype_template.dir/build.make
prototypes/template/protype_template: source/core/liblethe-core.a
prototypes/template/protype_template: source/solvers/liblethe-solvers.a
prototypes/template/protype_template: source/core/liblethe-core.a
prototypes/template/protype_template: /home/shahab/dealii/build/lib/libdeal_II.so.9.2.0-pre
prototypes/template/protype_template: /home/shahab/p4est/FAST/lib/libp4est.so
prototypes/template/protype_template: /home/shahab/p4est/FAST/lib/libsc.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempif08.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempi_ignore_tkr.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_mpifh.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/libz.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/librol.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libtempus.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libmuelu-adapters.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libmuelu-interface.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libmuelu.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libmuelu_lgn.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/liblocathyra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/liblocaepetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/liblocalapack.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libloca.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libnoxepetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libnoxlapack.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libnox.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libintrepid2.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libintrepid.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libteko.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libstratimikos.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libstratimikosbelos.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libstratimikosamesos2.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libstratimikosaztecoo.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libstratimikosamesos.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libstratimikosml.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libstratimikosifpack.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libifpack2-adapters.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libifpack2.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libanasazitpetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libModeLaplace.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libanasaziepetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libanasazi.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libamesos2.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libshylu_nodetacho.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libshylu_nodehts.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libbelosxpetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libbelostpetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libbelosepetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libbelos.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libml.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libifpack.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libzoltan2.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libpamgen_extras.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libpamgen.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libamesos.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libgaleri-xpetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libgaleri-epetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libaztecoo.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libisorropia.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libxpetra-sup.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libxpetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libthyratpetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libthyraepetraext.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libthyraepetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libthyracore.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libtrilinosss.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libtpetraext.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libtpetrainout.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libtpetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libkokkostsqr.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libtpetraclassiclinalg.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libtpetraclassicnodeapi.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libtpetraclassic.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libepetraext.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libtriutils.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libshards.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libzoltan.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libepetra.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libminitensor.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libsacado.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/librtop.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libkokkoskernels.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libteuchoskokkoscomm.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libteuchoskokkoscompat.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libteuchosremainder.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libteuchosnumerics.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libteuchoscomm.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libteuchosparameterlist.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libteuchosparser.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libteuchoscore.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libkokkosalgorithms.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libkokkoscontainers.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libkokkoscore.so
prototypes/template/protype_template: /home/shahab/share/trilinos/lib/libgtest.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/libarpack.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/libnetcdf.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/liblapack.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/libblas.so
prototypes/template/protype_template: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
prototypes/template/protype_template: prototypes/template/CMakeFiles/protype_template.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable protype_template"
	cd /home/shahab/Lethe_newversion/lethe/prototypes/template && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/protype_template.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
prototypes/template/CMakeFiles/protype_template.dir/build: prototypes/template/protype_template

.PHONY : prototypes/template/CMakeFiles/protype_template.dir/build

prototypes/template/CMakeFiles/protype_template.dir/requires: prototypes/template/CMakeFiles/protype_template.dir/prototype_template.cc.o.requires

.PHONY : prototypes/template/CMakeFiles/protype_template.dir/requires

prototypes/template/CMakeFiles/protype_template.dir/clean:
	cd /home/shahab/Lethe_newversion/lethe/prototypes/template && $(CMAKE_COMMAND) -P CMakeFiles/protype_template.dir/cmake_clean.cmake
.PHONY : prototypes/template/CMakeFiles/protype_template.dir/clean

prototypes/template/CMakeFiles/protype_template.dir/depend:
	cd /home/shahab/Lethe_newversion/lethe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/prototypes/template /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/prototypes/template /home/shahab/Lethe_newversion/lethe/prototypes/template/CMakeFiles/protype_template.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : prototypes/template/CMakeFiles/protype_template.dir/depend

