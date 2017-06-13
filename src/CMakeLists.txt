find_package(OpenCV REQUIRED)
set(doldetect_src	doldetect.cpp dbscan.cpp)

# -- dynamic library -- #
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library(doldetect SHARED ${doldetect_src})
set_target_properties(doldetect PROPERTIES VERSION ${DOLDETECT_TARGET_VERSION} SOVERSION ${DOLDETECT_TARGET_SOVERSION})
target_link_libraries(doldetect ${OpenCV_LIBS})
target_include_directories(doldetect PUBLIC ${DolDetect_INCLUDE_DIR})

# -- static library -- #
add_library(doldetect_static STATIC ${doldetect_src})
target_link_libraries(doldetect_static ${OpenCV_LIBS})
target_include_directories(doldetect_static PUBLIC ${DolDetect_INCLUDE_DIR})
