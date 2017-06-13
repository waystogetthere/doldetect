cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# -- DolDetect project -- #
project(DolDetect)

# -- DolDetect version -- #
set(DOLDETECT_TARGET_VERSION "1.0" CACHE STRING "logical version")
set(DOLDETECT_TARGET_SOVERSION "1.0" CACHE STRING "soname version")

# -- DolDetect variables -- #
set(DolDetect_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(DolDetect_SRC_DIR ${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src)
