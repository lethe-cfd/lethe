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

# Utility rule file for boundary_cells_and_faces.release.diff.

# Include the progress variables for this target.
include tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff.dir/progress.make

tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff: tests/dem/boundary_cells_and_faces.release/diff
	cd /home/shahab/Lethe_newversion/lethe/tests/dem && echo dem/boundary_cells_and_faces.release:\ BUILD\ successful.
	cd /home/shahab/Lethe_newversion/lethe/tests/dem && echo dem/boundary_cells_and_faces.release:\ RUN\ successful.
	cd /home/shahab/Lethe_newversion/lethe/tests/dem && echo dem/boundary_cells_and_faces.release:\ DIFF\ successful.
	cd /home/shahab/Lethe_newversion/lethe/tests/dem && echo dem/boundary_cells_and_faces.release:\ PASSED.

tests/dem/boundary_cells_and_faces.release/diff: tests/dem/boundary_cells_and_faces.release/output
tests/dem/boundary_cells_and_faces.release/diff: tests/dem/boundary_cells_and_faces.output
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating boundary_cells_and_faces.release/diff"
	cd /home/shahab/Lethe_newversion/lethe/tests/dem/boundary_cells_and_faces.release && sh /home/shahab/dealii/build/share/deal.II/scripts/run_test.sh diff dem/boundary_cells_and_faces.release /usr/bin/numdiff /home/shahab/Lethe_newversion/lethe/tests/dem/boundary_cells_and_faces.output /home/shahab/Lethe_newversion/lethe/tests/dem/boundary_cells_and_faces.release/boundary_cells_and_faces.release

tests/dem/boundary_cells_and_faces.release/output: tests/dem/boundary_cells_and_faces.release/boundary_cells_and_faces.release
tests/dem/boundary_cells_and_faces.release/output: /home/shahab/dealii/build/share/deal.II/scripts/normalize.pl
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating boundary_cells_and_faces.release/output"
	cd /home/shahab/Lethe_newversion/lethe/tests/dem/boundary_cells_and_faces.release && sh /home/shahab/dealii/build/share/deal.II/scripts/run_test.sh run dem/boundary_cells_and_faces.release /home/shahab/Lethe_newversion/lethe/tests/dem/boundary_cells_and_faces.release/boundary_cells_and_faces.release
	cd /home/shahab/Lethe_newversion/lethe/tests/dem/boundary_cells_and_faces.release && /usr/bin/perl -pi /home/shahab/dealii/build/share/deal.II/scripts/normalize.pl /home/shahab/Lethe_newversion/lethe/tests/dem/boundary_cells_and_faces.release/output

boundary_cells_and_faces.release.diff: tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff
boundary_cells_and_faces.release.diff: tests/dem/boundary_cells_and_faces.release/diff
boundary_cells_and_faces.release.diff: tests/dem/boundary_cells_and_faces.release/output
boundary_cells_and_faces.release.diff: tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff.dir/build.make

.PHONY : boundary_cells_and_faces.release.diff

# Rule to build all files generated by this target.
tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff.dir/build: boundary_cells_and_faces.release.diff

.PHONY : tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff.dir/build

tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff.dir/clean:
	cd /home/shahab/Lethe_newversion/lethe/tests/dem && $(CMAKE_COMMAND) -P CMakeFiles/boundary_cells_and_faces.release.diff.dir/cmake_clean.cmake
.PHONY : tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff.dir/clean

tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff.dir/depend:
	cd /home/shahab/Lethe_newversion/lethe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/tests/dem /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/tests/dem /home/shahab/Lethe_newversion/lethe/tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/dem/CMakeFiles/boundary_cells_and_faces.release.diff.dir/depend

