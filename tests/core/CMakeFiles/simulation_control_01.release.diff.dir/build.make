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

# Utility rule file for simulation_control_01.release.diff.

# Include the progress variables for this target.
include tests/core/CMakeFiles/simulation_control_01.release.diff.dir/progress.make

tests/core/CMakeFiles/simulation_control_01.release.diff: tests/core/simulation_control_01.release/diff
	cd /home/shahab/Lethe_newversion/lethe/tests/core && echo core/simulation_control_01.release:\ BUILD\ successful.
	cd /home/shahab/Lethe_newversion/lethe/tests/core && echo core/simulation_control_01.release:\ RUN\ successful.
	cd /home/shahab/Lethe_newversion/lethe/tests/core && echo core/simulation_control_01.release:\ DIFF\ successful.
	cd /home/shahab/Lethe_newversion/lethe/tests/core && echo core/simulation_control_01.release:\ PASSED.

tests/core/simulation_control_01.release/diff: tests/core/simulation_control_01.release/output
tests/core/simulation_control_01.release/diff: tests/core/simulation_control_01.output
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating simulation_control_01.release/diff"
	cd /home/shahab/Lethe_newversion/lethe/tests/core/simulation_control_01.release && sh /home/shahab/dealii/build/share/deal.II/scripts/run_test.sh diff core/simulation_control_01.release /usr/bin/numdiff /home/shahab/Lethe_newversion/lethe/tests/core/simulation_control_01.output /home/shahab/Lethe_newversion/lethe/tests/core/simulation_control_01.release/simulation_control_01.release

tests/core/simulation_control_01.release/output: tests/core/simulation_control_01.release/simulation_control_01.release
tests/core/simulation_control_01.release/output: /home/shahab/dealii/build/share/deal.II/scripts/normalize.pl
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating simulation_control_01.release/output"
	cd /home/shahab/Lethe_newversion/lethe/tests/core/simulation_control_01.release && sh /home/shahab/dealii/build/share/deal.II/scripts/run_test.sh run core/simulation_control_01.release /home/shahab/Lethe_newversion/lethe/tests/core/simulation_control_01.release/simulation_control_01.release
	cd /home/shahab/Lethe_newversion/lethe/tests/core/simulation_control_01.release && /usr/bin/perl -pi /home/shahab/dealii/build/share/deal.II/scripts/normalize.pl /home/shahab/Lethe_newversion/lethe/tests/core/simulation_control_01.release/output

simulation_control_01.release.diff: tests/core/CMakeFiles/simulation_control_01.release.diff
simulation_control_01.release.diff: tests/core/simulation_control_01.release/diff
simulation_control_01.release.diff: tests/core/simulation_control_01.release/output
simulation_control_01.release.diff: tests/core/CMakeFiles/simulation_control_01.release.diff.dir/build.make

.PHONY : simulation_control_01.release.diff

# Rule to build all files generated by this target.
tests/core/CMakeFiles/simulation_control_01.release.diff.dir/build: simulation_control_01.release.diff

.PHONY : tests/core/CMakeFiles/simulation_control_01.release.diff.dir/build

tests/core/CMakeFiles/simulation_control_01.release.diff.dir/clean:
	cd /home/shahab/Lethe_newversion/lethe/tests/core && $(CMAKE_COMMAND) -P CMakeFiles/simulation_control_01.release.diff.dir/cmake_clean.cmake
.PHONY : tests/core/CMakeFiles/simulation_control_01.release.diff.dir/clean

tests/core/CMakeFiles/simulation_control_01.release.diff.dir/depend:
	cd /home/shahab/Lethe_newversion/lethe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/tests/core /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/tests/core /home/shahab/Lethe_newversion/lethe/tests/core/CMakeFiles/simulation_control_01.release.diff.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/core/CMakeFiles/simulation_control_01.release.diff.dir/depend

