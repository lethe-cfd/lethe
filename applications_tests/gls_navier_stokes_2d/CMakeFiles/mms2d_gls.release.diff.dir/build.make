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

# Utility rule file for mms2d_gls.release.diff.

# Include the progress variables for this target.
include applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff.dir/progress.make

applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff: applications_tests/gls_navier_stokes_2d/mms2d_gls.release/diff
	cd /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d && echo gls_navier_stokes_2d/mms2d_gls.release:\ BUILD\ successful.
	cd /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d && echo gls_navier_stokes_2d/mms2d_gls.release:\ RUN\ successful.
	cd /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d && echo gls_navier_stokes_2d/mms2d_gls.release:\ DIFF\ successful.
	cd /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d && echo gls_navier_stokes_2d/mms2d_gls.release:\ PASSED.

applications_tests/gls_navier_stokes_2d/mms2d_gls.release/diff: applications_tests/gls_navier_stokes_2d/mms2d_gls.release/output
applications_tests/gls_navier_stokes_2d/mms2d_gls.release/diff: applications_tests/gls_navier_stokes_2d/mms2d_gls.output
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating mms2d_gls.release/diff"
	cd /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d/mms2d_gls.release && sh /home/shahab/dealii/build/share/deal.II/scripts/run_test.sh diff gls_navier_stokes_2d/mms2d_gls.release /usr/bin/numdiff /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d/mms2d_gls.output /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d/gls_navier_stokes_2d /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d/mms2d_gls.prm

applications_tests/gls_navier_stokes_2d/mms2d_gls.release/output: applications/gls_navier_stokes_2d/gls_navier_stokes_2d
applications_tests/gls_navier_stokes_2d/mms2d_gls.release/output: /home/shahab/dealii/build/share/deal.II/scripts/normalize.pl
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating mms2d_gls.release/output"
	cd /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d/mms2d_gls.release && sh /home/shahab/dealii/build/share/deal.II/scripts/run_test.sh run gls_navier_stokes_2d/mms2d_gls.release /home/shahab/Lethe_newversion/lethe/applications/gls_navier_stokes_2d/gls_navier_stokes_2d /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d/mms2d_gls.prm
	cd /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d/mms2d_gls.release && /usr/bin/perl -pi /home/shahab/dealii/build/share/deal.II/scripts/normalize.pl /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d/mms2d_gls.release/output

mms2d_gls.release.diff: applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff
mms2d_gls.release.diff: applications_tests/gls_navier_stokes_2d/mms2d_gls.release/diff
mms2d_gls.release.diff: applications_tests/gls_navier_stokes_2d/mms2d_gls.release/output
mms2d_gls.release.diff: applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff.dir/build.make

.PHONY : mms2d_gls.release.diff

# Rule to build all files generated by this target.
applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff.dir/build: mms2d_gls.release.diff

.PHONY : applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff.dir/build

applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff.dir/clean:
	cd /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d && $(CMAKE_COMMAND) -P CMakeFiles/mms2d_gls.release.diff.dir/cmake_clean.cmake
.PHONY : applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff.dir/clean

applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff.dir/depend:
	cd /home/shahab/Lethe_newversion/lethe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d /home/shahab/Lethe_newversion/lethe/applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : applications_tests/gls_navier_stokes_2d/CMakeFiles/mms2d_gls.release.diff.dir/depend

