
# Set the name of the project and target:
SET(TARGET "lethe-rpt")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#    FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#    FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#    SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC}) 
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.

FILE(GLOB_RECURSE TARGET_SRC *.cc)
FILE(GLOB_RECURSE TARGET_INC ${CMAKE_SOURCE_DIR}/include/core/*.h)
FILE(GLOB_RECURSE TARGET_INC ${CMAKE_SOURCE_DIR}/include/rpt/*.h)
SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC}) 

# Usually, you will not need to modify anything beyond this point...

ADD_LIBRARY(${TARGET}
  ${TARGET_SRC}
  )
DEAL_II_SETUP_TARGET(${TARGET})
TARGET_LINK_LIBRARIES(lethe-rpt lethe-core)
install(TARGETS lethe-rpt ARCHIVE DESTINATION lib)

