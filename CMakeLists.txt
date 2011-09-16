set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# Following block can broke build in case of cross-compilng
# but CMAKE_CROSSCOMPILING variable will be set only on project(OpenCV) command
# so we will try to detect crosscompiling by presense of CMAKE_TOOLCHAIN_FILE
if(NOT CMAKE_TOOLCHAIN_FILE)
# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
	if(UNIX)
		set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /lib /usr/lib)
	endif()

# it _must_ go before PROJECT(OpenCV) in order to work
	if(WIN32)
		set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
	else()
		set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation Directory")
	endif()

	if(MSVC)
		set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
	endif()  
endif(NOT CMAKE_TOOLCHAIN_FILE)

project(stitching)

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    )

set(stitching_libs opencv_core opencv_imgproc opencv_highgui opencv_features2d opencv_calib3d opencv_gpu)

FILE(GLOB stitching_files "*.hpp" "*.cpp")

set(the_target opencv_stitching)    
add_executable(${the_target} ${stitching_files})

add_dependencies(${the_target} ${stitching_libs})
set_target_properties(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      INSTALL_NAME_DIR lib
                      OUTPUT_NAME "opencv_stitching")
					  
if(ENABLE_SOLUTION_FOLDERS)
	set_target_properties(${the_target} PROPERTIES FOLDER "applications")
endif()	

target_link_libraries(${the_target} ${stitching_libs})

install(TARGETS ${the_target} RUNTIME DESTINATION bin COMPONENT main)
