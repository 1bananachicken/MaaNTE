cmake_minimum_required(VERSION 3.28)

project(maante-cpp-navi LANGUAGES CXX)

include(MaaUtils/MaaUtils.cmake)

set(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../deps")
if(NOT EXISTS "${DEPS_DIR}")
    message(FATAL_ERROR "MaaFramework dependencies not found: ${DEPS_DIR}")
endif()

set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO RelWithDebInfo)
set(CMAKE_MAP_IMPORTED_CONFIG_RELEASE RelWithDebInfo)
set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG RelWithDebInfo)
set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL RelWithDebInfo)

set(MaaFramework_DIR "${DEPS_DIR}/share/cmake/MaaFramework")
set(MaaAgentServer_DIR "${DEPS_DIR}/share/cmake/MaaAgentServer")
find_package(MaaFramework CONFIG REQUIRED)
find_package(MaaAgentServer CONFIG REQUIRED)
find_package(Threads REQUIRED)

file(GLOB_RECURSE NAVI_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.h")
add_executable(cpp-navi ${NAVI_SOURCES})
target_include_directories(cpp-navi PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(cpp-navi PRIVATE _CRT_SECURE_NO_WARNINGS)
set_target_properties(cpp-navi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${CONFIG}" CONFIG_UPPER)
    set_target_properties(cpp-navi PROPERTIES "RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}" "${CMAKE_CURRENT_SOURCE_DIR}/bin")
endforeach()
target_link_libraries(cpp-navi PRIVATE
    MaaUtils
    MaaFramework::MaaFramework
    MaaAgentServer::MaaAgentServer
    Boost::system
    ${OpenCV_LIBS}
    ONNXRuntime::ONNXRuntime
    Threads::Threads
)

if(WIN32)
    target_link_libraries(cpp-navi PRIVATE ws2_32)
endif()

add_dependencies(cpp-navi MaaUtils)

if(WIN32)
    file(GLOB MAADEPS_RUNTIME_DLLS "${MAADEPS_DIR}/runtime/${MAADEPS_TRIPLET}/*.dll")
    add_custom_command(TARGET cpp-navi POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:cpp-navi>/plugins"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEPS_DIR}/bin/MaaFramework.dll"
            "${DEPS_DIR}/bin/MaaAgentServer.dll"
            "$<TARGET_FILE_DIR:cpp-navi>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MAADEPS_RUNTIME_DLLS} "$<TARGET_FILE_DIR:cpp-navi>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:MaaUtils>" "$<TARGET_FILE_DIR:cpp-navi>"
        VERBATIM
    )
endif()

install(TARGETS cpp-navi RUNTIME DESTINATION agent)
if(WIN32)
    install(FILES $<TARGET_PDB_FILE:cpp-navi> DESTINATION agent OPTIONAL)
endif()

option(NAVI_BUILD_TESTS "Build C++ Navi unit tests" ON)
if(NAVI_BUILD_TESTS)
    add_executable(cpp-navi-tests
        tests/navi_tests.cpp
        src/coordinate.cpp
        src/route.cpp
        src/util.cpp
        src/websocket_service.cpp
    )
    target_include_directories(cpp-navi-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(cpp-navi-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(cpp-navi-tests PRIVATE
        MaaUtils
        MaaFramework::MaaFramework
        Boost::system
        ${OpenCV_LIBS}
        Threads::Threads
    )
    if(WIN32)
        target_link_libraries(cpp-navi-tests PRIVATE ws2_32)
    endif()
    add_custom_command(TARGET cpp-navi-tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:cpp-navi-tests>/plugins"
        VERBATIM
    )
endif()
