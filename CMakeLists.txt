cmake_minimum_required(VERSION 3.16)
project(treesem_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KAMA_BUILD_TREESEM_SERVER "Build the treeSem backend entrypoint" ON)
option(KAMA_BUILD_GOMOKU_SERVER "Build the original Gomoku example" OFF)
option(KAMA_BUILD_TESTS "Build lightweight HTTP core tests" ON)

set(MUDUO_ROOT "" CACHE PATH "Muduo installation prefix")
set(NLOHMANN_JSON_ROOT "" CACHE PATH "nlohmann/json installation prefix")

find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_path(NLOHMANN_JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    HINTS
        ${NLOHMANN_JSON_ROOT}/include
        $ENV{CONDA_PREFIX}/include
)
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR
        "nlohmann/json was not found. Configure with "
        "-DNLOHMANN_JSON_ROOT=/path/to/prefix")
endif()

find_path(MUDUO_INCLUDE_DIR
    NAMES muduo/net/TcpServer.h
    HINTS ${MUDUO_ROOT}/include
)
find_library(MUDUO_NET_LIBRARY
    NAMES muduo_net
    HINTS ${MUDUO_ROOT}/lib ${MUDUO_ROOT}/lib64
)
find_library(MUDUO_BASE_LIBRARY
    NAMES muduo_base
    HINTS ${MUDUO_ROOT}/lib ${MUDUO_ROOT}/lib64
)

if(NOT MUDUO_INCLUDE_DIR OR NOT MUDUO_NET_LIBRARY OR NOT MUDUO_BASE_LIBRARY)
    message(FATAL_ERROR
        "Muduo was not found. Configure with "
        "-DMUDUO_ROOT=/path/to/muduo/install")
endif()

set(KAMA_HTTP_CORE_SOURCES
    HttpServer/src/http/HttpContext.cpp
    HttpServer/src/http/HttpRequest.cpp
    HttpServer/src/http/HttpResponse.cpp
    HttpServer/src/http/HttpServer.cpp
    HttpServer/src/middleware/MiddlewareChain.cpp
    HttpServer/src/middleware/cors/CorsMiddleware.cpp
    HttpServer/src/router/Router.cpp
    HttpServer/src/session/Session.cpp
    HttpServer/src/session/SessionManager.cpp
    HttpServer/src/session/SessionStorage.cpp
    HttpServer/src/ssl/SslConfig.cpp
    HttpServer/src/ssl/SslConnection.cpp
    HttpServer/src/ssl/SslContext.cpp
)

add_library(kama_http_core STATIC ${KAMA_HTTP_CORE_SOURCES})
target_include_directories(kama_http_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/HttpServer/include
)
target_include_directories(kama_http_core SYSTEM PUBLIC ${MUDUO_INCLUDE_DIR})
target_link_libraries(kama_http_core
    PUBLIC
        ${MUDUO_NET_LIBRARY}
        ${MUDUO_BASE_LIBRARY}
        OpenSSL::SSL
        OpenSSL::Crypto
        Threads::Threads
)
target_compile_options(kama_http_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(treesem_runtime STATIC
    WebApps/TreeSemServer/src/client/PythonModelClient.cpp
    WebApps/TreeSemServer/src/concurrency/BoundedWorkerPool.cpp
    WebApps/TreeSemServer/src/service/InferenceScheduler.cpp
    WebApps/TreeSemServer/src/service/PythonModelProxy.cpp
)
target_include_directories(treesem_runtime
    PUBLIC
        ${PROJECT_SOURCE_DIR}/WebApps/TreeSemServer/include
)
target_link_libraries(treesem_runtime PUBLIC kama_http_core)
target_include_directories(treesem_runtime SYSTEM PUBLIC ${NLOHMANN_JSON_INCLUDE_DIR})
target_link_libraries(treesem_runtime PUBLIC CURL::libcurl)
target_compile_options(treesem_runtime PRIVATE -Wall -Wextra -Wpedantic)

if(KAMA_BUILD_TREESEM_SERVER)
    add_executable(treesem_server
        WebApps/TreeSemServer/src/main.cpp
    )
    target_link_libraries(treesem_server PRIVATE treesem_runtime)
endif()

if(KAMA_BUILD_GOMOKU_SERVER)
    find_path(MYSQL_INCLUDE_DIR
        NAMES mysql/jdbc.h
        HINTS /usr/include/mysql-cppconn-8 /usr/local/include
    )
    find_library(MYSQLCPPCONN_LIBRARY NAMES mysqlcppconn mysqlcppconn8)
    find_library(MYSQLCLIENT_LIBRARY NAMES mysqlclient)
    if(NOT MYSQL_INCLUDE_DIR OR NOT MYSQLCPPCONN_LIBRARY OR NOT MYSQLCLIENT_LIBRARY)
        message(FATAL_ERROR "MySQL client dependencies are required for Gomoku")
    endif()

    file(GLOB_RECURSE GOMOKU_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/WebApps/GomokuServer/src/*.cpp"
    )
    add_executable(gomoku_server
        ${GOMOKU_SOURCES}
        HttpServer/src/utils/db/DbConnection.cpp
        HttpServer/src/utils/db/DbConnectionPool.cpp
    )
    target_include_directories(gomoku_server PRIVATE
        ${PROJECT_SOURCE_DIR}/WebApps/GomokuServer/include
        ${MYSQL_INCLUDE_DIR}
    )
    target_link_libraries(gomoku_server PRIVATE
        kama_http_core
        ${MYSQLCPPCONN_LIBRARY}
        ${MYSQLCLIENT_LIBRARY}
    )
endif()

if(KAMA_BUILD_TESTS)
    enable_testing()

    add_executable(http_context_test tests/http_context_test.cpp)
    target_link_libraries(http_context_test PRIVATE kama_http_core)
    add_test(NAME http_context_test COMMAND http_context_test)

    add_executable(router_test tests/router_test.cpp)
    target_link_libraries(router_test PRIVATE kama_http_core)
    add_test(NAME router_test COMMAND router_test)

    add_executable(bounded_worker_pool_test tests/bounded_worker_pool_test.cpp)
    target_link_libraries(bounded_worker_pool_test PRIVATE treesem_runtime)
    add_test(NAME bounded_worker_pool_test COMMAND bounded_worker_pool_test)

    add_executable(inference_scheduler_test tests/inference_scheduler_test.cpp)
    target_link_libraries(inference_scheduler_test PRIVATE treesem_runtime)
    add_test(NAME inference_scheduler_test COMMAND inference_scheduler_test)

    add_executable(python_model_proxy_test tests/python_model_proxy_test.cpp)
    target_link_libraries(python_model_proxy_test PRIVATE treesem_runtime)
    add_test(NAME python_model_proxy_test COMMAND python_model_proxy_test)

    add_test(
        NAME python_adapter_contract_test
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s ${PROJECT_SOURCE_DIR}/PythonServices/TreeSemModelAdapter/tests
            -p test_server_contract.py
    )
    set_tests_properties(python_adapter_contract_test PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/PythonServices/TreeSemModelAdapter"
    )
endif()
