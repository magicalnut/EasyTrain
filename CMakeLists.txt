cmake_minimum_required(VERSION 3.19)
project(EasyTrain LANGUAGES CXX)

# libtorch 2.13 的头文件用到了 C++20 语法（指定初始化器、聚合体默认成员初始化器），
# 必须显式开 /std:c++20，否则报 C7555/C7582。qt_standard_project_setup() 不会自动提升标准。
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /DEBUG:NONE")
endif()

include(FetchContent)

FetchContent_Declare(
    opencv
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opencv
)

set(BUILD_SHARED_LIBS ON)
set(BUILD_TESTS OFF)
set(BUILD_PERF_TESTS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_opencv_apps OFF)
set(BUILD_opencv_python OFF)
set(BUILD_opencv_java OFF)
set(BUILD_LIST "core;imgproc;imgcodecs;highgui")
set(BUILD_opencv_world ON)
set(WITH_QT OFF)  # 不用 OpenCV 的 Qt 窗口后端（自己用 cv::Mat + QLabel 渲染）。
                  # 否则 opencv_world 会依赖 Qt6Test/OpenGL/OpenGLWidgets/Concurrent/PrintSupport 等一堆用不到的 Qt 模块
set(WITH_OPENGL ON)
set(OPENCV_PYTHON_DISABLE ON)
set(WITH_IPP OFF)
set(WITH_FFMPEG OFF)

FetchContent_MakeAvailable(opencv)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets CoreTools)

set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

qt_standard_project_setup()

qt_add_executable(EasyTrain
    WIN32 MACOSX_BUNDLE
    main.cpp
    mainwindow.h mainwindow.cpp
    monitor.h monitor.cpp
    plot_widget.h plot_widget.cpp
    model/layer_config.h
    model/shape.h
    model/shape.cpp
    model/layer_factory.h
    model/layer_factory.cpp
    model/dag_model.h
    model/test.h model/test.cpp


)

target_link_libraries(EasyTrain
    PRIVATE
        Qt::Core
        Qt::Widgets
        opencv_world
        ${TORCH_LIBRARIES}
)

# OpenCV 由 FetchContent 源码编译，opencv_world 目标没有把各模块的源码头目录
# 传播到本 target（编译期 C1083 找不到 opencv2/core.hpp），这里显式补齐：
#   各模块源码头 + 顶层 umbrella 头 + 构建生成的 cvconfig.h / opencv_modules.hpp
target_include_directories(EasyTrain PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/imgproc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/imgcodecs/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/highgui/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/world/include
    ${CMAKE_BINARY_DIR}
)

# 运行期 Windows 只从 exe 同目录 + PATH 找 DLL。opencv_world 的 DLL 生成在 opencv 构建目录、
# libtorch 的 DLL 在 libtorch/lib，都不在 exe 目录也不在 PATH 上，直接运行会报"找不到 XXX.dll"。
# 这里在每次构建后把这两套 DLL 复制到 exe 旁边（Qt 自身的 DLL 已由 Qt Creator 运行环境 PATH 提供）。
file(GLOB TORCH_RUNTIME_DLLS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/*.dll")
add_custom_command(TARGET EasyTrain POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:opencv_world>
        $<TARGET_FILE_DIR:EasyTrain>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_RUNTIME_DLLS}
        $<TARGET_FILE_DIR:EasyTrain>
    COMMENT "Copying OpenCV + LibTorch runtime DLLs next to EasyTrain.exe"
    VERBATIM
)

# Qt 自身的 DLL + 平台/图片等插件也部署到 exe 旁边，使 exe 脱离 Qt Creator 的 PATH 也能独立运行。
# 用官方 windeployqt（而非手动复制 Qt6Core/Gui/Widgets.dll）：它会一并带上 platforms/qwindows.dll
# 等插件，缺失这些插件 Qt 应用会直接报"could not find or load the Qt platform plugin"。
# 放在 opencv/torch 复制之后执行，windeployqt 看到它们已在 exe 目录会跳过，不会重复部署。
add_custom_command(TARGET EasyTrain POST_BUILD
    COMMAND Qt6::windeployqt --no-translations "$<TARGET_FILE:EasyTrain>"
    COMMENT "Deploying Qt runtime + plugins next to EasyTrain.exe"
    VERBATIM
)

install(TARGETS EasyTrain
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(WIN32)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/"
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            FILES_MATCHING PATTERN "*.dll")
elseif(UNIX)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            FILES_MATCHING PATTERN "*.so*")
endif()

qt_generate_deploy_app_script(
    TARGET EasyTrain
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
install(FILES $<TARGET_FILE:opencv_world> DESTINATION ${CMAKE_INSTALL_BINDIR})
