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

# Utility rule file for particle_particle_contact_force_linear.debug.diff.

# Include the progress variables for this target.
include tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff.dir/progress.make

tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff: tests/dem/particle_particle_contact_force_linear.debug/diff
	cd /home/shahab/Lethe_newversion/lethe/tests/dem && echo dem/particle_particle_contact_force_linear.debug:\ BUILD\ successful.
	cd /home/shahab/Lethe_newversion/lethe/tests/dem && echo dem/particle_particle_contact_force_linear.debug:\ RUN\ successful.
	cd /home/shahab/Lethe_newversion/lethe/tests/dem && echo dem/particle_particle_contact_force_linear.debug:\ DIFF\ successful.
	cd /home/shahab/Lethe_newversion/lethe/tests/dem && echo dem/particle_particle_contact_force_linear.debug:\ PASSED.

tests/dem/particle_particle_contact_force_linear.debug/diff: tests/dem/particle_particle_contact_force_linear.debug/output
tests/dem/particle_particle_contact_force_linear.debug/diff: tests/dem/particle_particle_contact_force_linear.output
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating particle_particle_contact_force_linear.debug/diff"
	cd /home/shahab/Lethe_newversion/lethe/tests/dem/particle_particle_contact_force_linear.debug && sh /home/shahab/dealii/build/share/deal.II/scripts/run_test.sh diff dem/particle_particle_contact_force_linear.debug /usr/bin/numdiff /home/shahab/Lethe_newversion/lethe/tests/dem/particle_particle_contact_force_linear.output /home/shahab/Lethe_newversion/lethe/tests/dem/particle_particle_contact_force_linear.debug/particle_particle_contact_force_linear.debug

tests/dem/particle_particle_contact_force_linear.debug/output: tests/dem/particle_particle_contact_force_linear.debug/particle_particle_contact_force_linear.debug
tests/dem/particle_particle_contact_force_linear.debug/output: /home/shahab/dealii/build/share/deal.II/scripts/normalize.pl
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/shahab/Lethe_newversion/lethe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating particle_particle_contact_force_linear.debug/output"
	cd /home/shahab/Lethe_newversion/lethe/tests/dem/particle_particle_contact_force_linear.debug && sh /home/shahab/dealii/build/share/deal.II/scripts/run_test.sh run dem/particle_particle_contact_force_linear.debug /home/shahab/Lethe_newversion/lethe/tests/dem/particle_particle_contact_force_linear.debug/particle_particle_contact_force_linear.debug
	cd /home/shahab/Lethe_newversion/lethe/tests/dem/particle_particle_contact_force_linear.debug && /usr/bin/perl -pi /home/shahab/dealii/build/share/deal.II/scripts/normalize.pl /home/shahab/Lethe_newversion/lethe/tests/dem/particle_particle_contact_force_linear.debug/output

particle_particle_contact_force_linear.debug.diff: tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff
particle_particle_contact_force_linear.debug.diff: tests/dem/particle_particle_contact_force_linear.debug/diff
particle_particle_contact_force_linear.debug.diff: tests/dem/particle_particle_contact_force_linear.debug/output
particle_particle_contact_force_linear.debug.diff: tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff.dir/build.make

.PHONY : particle_particle_contact_force_linear.debug.diff

# Rule to build all files generated by this target.
tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff.dir/build: particle_particle_contact_force_linear.debug.diff

.PHONY : tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff.dir/build

tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff.dir/clean:
	cd /home/shahab/Lethe_newversion/lethe/tests/dem && $(CMAKE_COMMAND) -P CMakeFiles/particle_particle_contact_force_linear.debug.diff.dir/cmake_clean.cmake
.PHONY : tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff.dir/clean

tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff.dir/depend:
	cd /home/shahab/Lethe_newversion/lethe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/tests/dem /home/shahab/Lethe_newversion/lethe /home/shahab/Lethe_newversion/lethe/tests/dem /home/shahab/Lethe_newversion/lethe/tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/dem/CMakeFiles/particle_particle_contact_force_linear.debug.diff.dir/depend

