# CMake script to configure and build a test project

set(TEST_ARG_LIST)

# Expect actual arguments to start at index 3 (cmake -P <script_name>)
foreach(ARG_INDEX RANGE 3 ${CMAKE_ARGC})
    list(APPEND TEST_ARG_LIST "${CMAKE_ARGV${ARG_INDEX}}")
endforeach()

set(options "USE_INSTALLED_CORROSION")
set(oneValueArgs
    SOURCE_DIR
    BINARY_DIR
    GENERATOR
    GENERATOR_PLATFORM
    RUST_TOOLCHAIN
    CARGO_TARGET
    C_COMPILER
    CXX_COMPILER
    C_COMPILER_TARGET
    CXX_COMPILER_TARGET
    SYSTEM_NAME
    CARGO_PROFILE
    OSX_ARCHITECTURES
    TOOLCHAIN_FILE
)
set(multiValueArgs "PASS_THROUGH_ARGS")
cmake_parse_arguments(TEST "${options}" "${oneValueArgs}"
                      "${multiValueArgs}" ${TEST_ARG_LIST} )

set(configure_args "")
if(TEST_CARGO_TARGET)
    list(APPEND configure_args "-DRust_CARGO_TARGET=${TEST_CARGO_TARGET}")
endif()
if(TEST_USE_INSTALLED_CORROSION)
    list(APPEND configure_args "-DCORROSION_TESTS_FIND_CORROSION=ON")
endif()
if(TEST_GENERATOR_PLATFORM)
    list(APPEND configure_args "-A${TEST_GENERATOR_PLATFORM}")
endif()
if(TEST_C_COMPILER)
    list(APPEND configure_args "-DCMAKE_C_COMPILER=${TEST_C_COMPILER}")
endif()
if(TEST_CXX_COMPILER)
    list(APPEND configure_args "-DCMAKE_CXX_COMPILER=${TEST_CXX_COMPILER}")
endif()
if(TEST_C_COMPILER_TARGET)
    list(APPEND configure_args "-DCMAKE_C_COMPILER_TARGET=${TEST_C_COMPILER_TARGET}")
endif()
if(TEST_CXX_COMPILER_TARGET)
    list(APPEND configure_args "-DCMAKE_CXX_COMPILER_TARGET=${TEST_CXX_COMPILER_TARGET}")
endif()
if(TEST_SYSTEM_NAME)
    list(APPEND configure_args "-DCMAKE_SYSTEM_NAME=${TEST_SYSTEM_NAME}")
endif()
if(TEST_OSX_ARCHITECTURES)
    list(APPEND configure_args "-DCMAKE_OSX_ARCHITECTURES=${TEST_OSX_ARCHITECTURES}")
endif()
if(TEST_TOOLCHAIN_FILE)
    list(APPEND configure_args "-DCMAKE_TOOLCHAIN_FILE=${TEST_TOOLCHAIN_FILE}")
endif()
if(TEST_CARGO_PROFILE)
    list(APPEND configure_args "-DCARGO_PROFILE=${TEST_CARGO_PROFILE}")
endif()

# Remove old binary directory
file(REMOVE_RECURSE "${TEST_BINARY_DIR}")

file(MAKE_DIRECTORY "${TEST_BINARY_DIR}")

message(STATUS "TEST_BINARY_DIRECTORY: ${TEST_BINARY_DIR}")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
            "-G${TEST_GENERATOR}"
            "-DRust_TOOLCHAIN=${TEST_RUST_TOOLCHAIN}"
            --log-level Debug
            ${configure_args}
            ${TEST_PASS_THROUGH_ARGS}
            -S "${TEST_SOURCE_DIR}"
            -B "${TEST_BINARY_DIR}"
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE EXIT_CODE
)

if (NOT "${EXIT_CODE}" EQUAL 0)
    message(FATAL_ERROR "Configure step failed. Exit code: `${EXIT_CODE}`")
endif()

if ("${TEST_GENERATOR}" STREQUAL "Ninja Multi-Config"
        OR "${TEST_GENERATOR}" MATCHES "Visual Studio"
    )
    foreach(config Debug Release RelWithDebInfo)
        execute_process(
                COMMAND "${CMAKE_COMMAND}"
                    --build "${TEST_BINARY_DIR}"
                    --config "${config}"
                COMMAND_ECHO STDOUT
                RESULT_VARIABLE EXIT_CODE
        )
        if (NOT "${EXIT_CODE}" EQUAL 0)
            message(FATAL_ERROR "Build step failed for config `${config}`. "
                    "Exit code: `${EXIT_CODE}`")
        endif()
    endforeach()
else()
    execute_process(
            COMMAND "${CMAKE_COMMAND}" --build "${TEST_BINARY_DIR}"
            COMMAND_ECHO STDOUT
            RESULT_VARIABLE EXIT_CODE
    )
    if (NOT "${EXIT_CODE}" EQUAL 0)
        message(FATAL_ERROR "Build step failed. Exit code: `${EXIT_CODE}`")
    endif()
endif()


