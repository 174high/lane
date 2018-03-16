# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)
# 项目信息
project (Lane)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

set(CMAKE_CXX_STANDARD 11)

# 编译模式
#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(LINKER_LIBS "")

# opencv
find_package(OpenCV REQUIRED)
list(APPEND LINKER_LIBS ${OpenCV_LIBRARIES})

# gflags
find_package(GFlags REQUIRED)
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})  
list(APPEND LINKER_LIBS ${GFLAGS_LIBRARIES})

message(${LINKER_LIBS})

# 指定生成目标
add_executable(gen_diff_mean src/gen_diff_mean.cpp)
target_link_libraries(gen_diff_mean ${LINKER_LIBS})

add_executable(gen_lane src/gen_lane.cpp)
target_link_libraries(gen_lane ${LINKER_LIBS})

add_executable(lan_env src/lan_env.cpp)
target_link_libraries(lan_env ${LINKER_LIBS})

add_executable(mouse_crop src/mouse_crop.cpp)
target_link_libraries(mouse_crop ${LINKER_LIBS})

add_executable(rgb2lab src/rgb2lab.cpp)
target_link_libraries(rgb2lab ${LINKER_LIBS})

add_executable(th1 src/th1.cpp)
target_link_libraries(th1 ${LINKER_LIBS})
