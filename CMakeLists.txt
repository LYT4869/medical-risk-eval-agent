cmake_minimum_required(VERSION 3.16)
project(treesem_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KAMA_BUILD_TREESEM_SERVER "Build the treeSem backend entrypoint" ON)
option(KAMA_BUILD_GOMOKU_SERVER "Build the original Gomoku example" OFF)
option(KAMA_BUILD_TESTS "Build lightweight HTTP core tests" ON)
option(KAMA_ENABLE_ONNXRUNTIME "Build the C++ ONNX Runtime model backend" ON)
option(KAMA_ENABLE_MYSQL "Build the treeSem MySQL persistence backend" ON)
option(KAMA_ENABLE_AUTH "Build treeSem authentication and Argon2id support" ON)
option(TREESEM_ENABLE_MYSQL_INTEGRATION_TESTS
    "Register tests that require an externally managed MySQL 8 test database" OFF)

set(MUDUO_ROOT "" CACHE PATH "Muduo installation prefix")
set(NLOHMANN_JSON_ROOT "" CACHE PATH "nlohmann/json installation prefix")
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime C++ installation prefix")
set(MYSQL_CONNECTOR_CPP_ROOT "" CACHE PATH "MySQL Connector/C++ installation prefix")
set(TREESEM_MODEL_PYTHON_EXECUTABLE "" CACHE FILEPATH
    "Python executable with torch/sklearn for real model tests")
set(TREESEM_TEST_BUNDLE_DIR "" CACHE PATH "Existing PPH serving bundle for tests")
set(TREESEM_TEST_ARTIFACT "" CACHE FILEPATH "Trusted PPH artifact for exporter tests")
set(TREESEM_TEST_RAW_FILE "" CACHE FILEPATH "Raw PPH CSV for exporter tests")

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
    HttpServer/src/http/AsyncHttp.cpp
    HttpServer/src/http/HttpContext.cpp
    HttpServer/src/http/HttpRequest.cpp
    HttpServer/src/http/HttpResponse.cpp
    HttpServer/src/http/HttpServer.cpp
    HttpServer/src/middleware/MiddlewareChain.cpp
    HttpServer/src/middleware/ObservabilityMiddleware.cpp
    HttpServer/src/middleware/cors/CorsMiddleware.cpp
    HttpServer/src/observability/MetricsRegistry.cpp
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
    WebApps/TreeSemServer/src/api/ApiException.cpp
    WebApps/TreeSemServer/src/api/HealthController.cpp
    WebApps/TreeSemServer/src/api/MetricsController.cpp
    WebApps/TreeSemServer/src/api/HttpErrorMapper.cpp
    WebApps/TreeSemServer/src/api/PredictionController.cpp
    WebApps/TreeSemServer/src/api/PredictionJsonCodec.cpp
    WebApps/TreeSemServer/src/api/BusinessJsonCodec.cpp
    WebApps/TreeSemServer/src/api/BusinessController.cpp
    WebApps/TreeSemServer/src/api/ChatController.cpp
    WebApps/TreeSemServer/src/api/ChatJsonCodec.cpp
    WebApps/TreeSemServer/src/api/AuthController.cpp
    WebApps/TreeSemServer/src/api/DoctorController.cpp
    WebApps/TreeSemServer/src/application/PredictionService.cpp
    WebApps/TreeSemServer/src/application/BusinessException.cpp
    WebApps/TreeSemServer/src/application/SessionService.cpp
    WebApps/TreeSemServer/src/application/RecordServices.cpp
    WebApps/TreeSemServer/src/application/AgentApplicationService.cpp
    WebApps/TreeSemServer/src/application/AuthService.cpp
    WebApps/TreeSemServer/src/application/AuditService.cpp
    WebApps/TreeSemServer/src/client/PythonModelClient.cpp
    WebApps/TreeSemServer/src/client/PythonAgentClient.cpp
    WebApps/TreeSemServer/src/config/TreeSemServerConfig.cpp
    WebApps/TreeSemServer/src/concurrency/BoundedWorkerPool.cpp
    WebApps/TreeSemServer/src/infrastructure/model/RemoteTreeSemModelService.cpp
    WebApps/TreeSemServer/src/infrastructure/model/FeaturePreprocessor.cpp
    WebApps/TreeSemServer/src/infrastructure/model/NativeDecisionTree.cpp
    WebApps/TreeSemServer/src/infrastructure/model/RoutingModelServices.cpp
    WebApps/TreeSemServer/src/infrastructure/model/ServingBundle.cpp
    WebApps/TreeSemServer/src/infrastructure/process/ProcessSignalHandler.cpp
    WebApps/TreeSemServer/src/infrastructure/persistence/InMemoryTreeSemStore.cpp
    WebApps/TreeSemServer/src/infrastructure/support/ValueSupport.cpp
    WebApps/TreeSemServer/src/domain/BusinessTypes.cpp
    WebApps/TreeSemServer/src/domain/SecurityTypes.cpp
    WebApps/TreeSemServer/src/security/JwtService.cpp
    WebApps/TreeSemServer/src/security/SecurityMiddleware.cpp
    WebApps/TreeSemServer/src/serialization/ModelResultJsonCodec.cpp
    WebApps/TreeSemServer/src/model/ModelException.cpp
    WebApps/TreeSemServer/src/service/InferenceScheduler.cpp
    WebApps/TreeSemServer/src/service/BlockingTaskScheduler.cpp
)
target_include_directories(treesem_runtime
    PUBLIC
        ${PROJECT_SOURCE_DIR}/WebApps/TreeSemServer/include
)
target_link_libraries(treesem_runtime PUBLIC kama_http_core)
target_include_directories(treesem_runtime SYSTEM PUBLIC ${NLOHMANN_JSON_INCLUDE_DIR})
target_link_libraries(treesem_runtime PUBLIC CURL::libcurl)
target_compile_options(treesem_runtime PRIVATE -Wall -Wextra -Wpedantic)

if(KAMA_ENABLE_AUTH)
    find_library(ARGON2_LIBRARY NAMES argon2 libargon2.so.1)
    if(NOT ARGON2_LIBRARY)
        message(FATAL_ERROR "libargon2 was not found; install libargon2-dev or disable KAMA_ENABLE_AUTH")
    endif()
    target_sources(treesem_runtime PRIVATE
        WebApps/TreeSemServer/src/security/PasswordHasher.cpp)
    target_link_libraries(treesem_runtime PUBLIC ${ARGON2_LIBRARY})
    target_compile_definitions(treesem_runtime PUBLIC TREESEM_HAS_AUTH=1)
endif()

if(KAMA_ENABLE_ONNXRUNTIME)
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_ROOT}/include
    )
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64
    )
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime C++ was not found. Configure ONNXRUNTIME_ROOT or "
            "use -DKAMA_ENABLE_ONNXRUNTIME=OFF for a remote-only build")
    endif()
    target_sources(treesem_runtime PRIVATE
        WebApps/TreeSemServer/src/infrastructure/model/OnnxTreeSemModelService.cpp
    )
    target_include_directories(treesem_runtime SYSTEM PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(treesem_runtime PUBLIC ${ONNXRUNTIME_LIBRARY})
    target_compile_definitions(treesem_runtime PUBLIC TREESEM_HAS_ONNXRUNTIME=1)
endif()

if(KAMA_ENABLE_MYSQL)
    find_path(MYSQL_CONNECTOR_CPP_INCLUDE_DIR
        NAMES mysql_connection.h
        HINTS ${MYSQL_CONNECTOR_CPP_ROOT}/include /usr/include
    )
    find_library(MYSQL_CONNECTOR_CPP_LIBRARY
        NAMES mysqlcppconn
        HINTS
            ${MYSQL_CONNECTOR_CPP_ROOT}/lib
            ${MYSQL_CONNECTOR_CPP_ROOT}/lib64
            ${MYSQL_CONNECTOR_CPP_ROOT}/lib/x86_64-linux-gnu
    )
    if(NOT MYSQL_CONNECTOR_CPP_INCLUDE_DIR OR NOT MYSQL_CONNECTOR_CPP_LIBRARY)
        message(FATAL_ERROR
            "MySQL Connector/C++ was not found. Install libmysqlcppconn-dev, "
            "configure MYSQL_CONNECTOR_CPP_ROOT, or use -DKAMA_ENABLE_MYSQL=OFF")
    endif()
    target_sources(treesem_runtime PRIVATE
        WebApps/TreeSemServer/src/infrastructure/persistence/MySqlConnectionPool.cpp
        WebApps/TreeSemServer/src/infrastructure/persistence/MySqlTreeSemStore.cpp
    )
    target_include_directories(treesem_runtime SYSTEM PUBLIC
        ${MYSQL_CONNECTOR_CPP_INCLUDE_DIR})
    target_link_libraries(treesem_runtime PUBLIC ${MYSQL_CONNECTOR_CPP_LIBRARY})
    target_compile_definitions(treesem_runtime PUBLIC TREESEM_HAS_MYSQL=1)
endif()

if(KAMA_BUILD_TREESEM_SERVER)
    add_executable(treesem_server
        WebApps/TreeSemServer/src/main.cpp
    )
    target_link_libraries(treesem_server PRIVATE treesem_runtime)
    target_compile_options(treesem_server PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(KAMA_BUILD_TREESEM_SERVER AND KAMA_ENABLE_MYSQL AND KAMA_ENABLE_AUTH)
    add_executable(treesem-admin
        WebApps/TreeSemServer/src/admin_main.cpp)
    target_link_libraries(treesem-admin PRIVATE treesem_runtime)
    target_compile_options(treesem-admin PRIVATE -Wall -Wextra -Wpedantic)
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

    add_executable(async_responder_test tests/async_responder_test.cpp)
    target_link_libraries(async_responder_test PRIVATE kama_http_core)
    add_test(NAME async_responder_test COMMAND async_responder_test)

    add_executable(observability_test tests/observability_test.cpp)
    target_link_libraries(observability_test PRIVATE treesem_runtime)
    add_test(NAME observability_test COMMAND observability_test)

    add_executable(bounded_worker_pool_test tests/bounded_worker_pool_test.cpp)
    target_link_libraries(bounded_worker_pool_test PRIVATE treesem_runtime)
    add_test(NAME bounded_worker_pool_test COMMAND bounded_worker_pool_test)

    add_executable(inference_scheduler_test tests/inference_scheduler_test.cpp)
    target_link_libraries(inference_scheduler_test PRIVATE treesem_runtime)
    add_test(NAME inference_scheduler_test COMMAND inference_scheduler_test)

    add_executable(prediction_json_codec_test tests/prediction_json_codec_test.cpp)
    target_link_libraries(prediction_json_codec_test PRIVATE treesem_runtime)
    add_test(NAME prediction_json_codec_test COMMAND prediction_json_codec_test)

    add_executable(remote_model_service_test tests/remote_model_service_test.cpp)
    target_link_libraries(remote_model_service_test PRIVATE treesem_runtime)
    add_test(NAME remote_model_service_test COMMAND remote_model_service_test)

    add_executable(routing_model_services_test tests/routing_model_services_test.cpp)
    target_link_libraries(routing_model_services_test PRIVATE treesem_runtime)
    add_test(NAME routing_model_services_test COMMAND routing_model_services_test)

    add_executable(prediction_controller_test tests/prediction_controller_test.cpp)
    target_link_libraries(prediction_controller_test PRIVATE treesem_runtime)
    add_test(NAME prediction_controller_test COMMAND prediction_controller_test)

    add_executable(server_config_test tests/server_config_test.cpp)
    target_link_libraries(server_config_test PRIVATE treesem_runtime)
    add_test(NAME server_config_test COMMAND server_config_test)

    add_executable(business_services_test tests/business_services_test.cpp)
    target_link_libraries(business_services_test PRIVATE treesem_runtime)
    add_test(NAME business_services_test COMMAND business_services_test)

    add_executable(business_json_codec_test tests/business_json_codec_test.cpp)
    target_link_libraries(business_json_codec_test PRIVATE treesem_runtime)
    add_test(NAME business_json_codec_test COMMAND business_json_codec_test)

    add_executable(agent_application_service_test tests/agent_application_service_test.cpp)
    target_link_libraries(agent_application_service_test PRIVATE treesem_runtime)
    add_test(NAME agent_application_service_test COMMAND agent_application_service_test)

    if(KAMA_ENABLE_AUTH)
        add_executable(security_core_test tests/security_core_test.cpp)
        target_link_libraries(security_core_test PRIVATE treesem_runtime)
        add_test(NAME security_core_test COMMAND security_core_test)
        set_tests_properties(security_core_test PROPERTIES TIMEOUT 30)
    endif()

    add_test(
        NAME python_adapter_contract_test
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s ${PROJECT_SOURCE_DIR}/PythonServices/TreeSemModelAdapter/tests
            -p test_server_contract.py
    )
    add_test(
        NAME python_agent_core_test
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s ${PROJECT_SOURCE_DIR}/PythonServices/TreeSemAgent/tests
            -p test_*.py
    )
    add_test(
        NAME python_knowledge_core_test
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s ${PROJECT_SOURCE_DIR}/PythonServices/TreeSemKnowledge/tests
            -p test_*.py
    )
    set_tests_properties(python_agent_core_test PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/PythonServices/TreeSemAgent"
    )
    set_tests_properties(python_knowledge_core_test PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/PythonServices/TreeSemKnowledge"
    )
    set_tests_properties(python_adapter_contract_test PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/PythonServices/TreeSemModelAdapter"
    )

    if(TREESEM_MODEL_PYTHON_EXECUTABLE
       AND TREESEM_TEST_BUNDLE_DIR
       AND TREESEM_TEST_ARTIFACT
       AND TREESEM_TEST_RAW_FILE)
        add_test(
            NAME python_bundle_runtime_test
            COMMAND ${TREESEM_MODEL_PYTHON_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_bundle_runtime.py
                --bundle ${TREESEM_TEST_BUNDLE_DIR}
                --artifact ${TREESEM_TEST_ARTIFACT}
                --raw-file ${TREESEM_TEST_RAW_FILE}
        )
        set_tests_properties(python_bundle_runtime_test PROPERTIES
            ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/PythonServices/TreeSemModelAdapter"
            TIMEOUT 120
        )
    endif()

    if(KAMA_ENABLE_ONNXRUNTIME AND TREESEM_TEST_BUNDLE_DIR)
        add_executable(onnx_model_service_test tests/onnx_model_service_test.cpp)
        target_link_libraries(onnx_model_service_test PRIVATE treesem_runtime)
        add_test(
            NAME onnx_model_service_test
            COMMAND onnx_model_service_test ${TREESEM_TEST_BUNDLE_DIR}
        )

        add_executable(onnx_predictions_dump tests/onnx_predictions_dump.cpp)
        target_link_libraries(onnx_predictions_dump PRIVATE treesem_runtime)
        if(TREESEM_MODEL_PYTHON_EXECUTABLE AND TREESEM_TEST_RAW_FILE)
            if(TREESEM_TEST_ARTIFACT)
                add_test(
                    NAME onnx_export_contract_test
                    COMMAND ${TREESEM_MODEL_PYTHON_EXECUTABLE}
                        ${PROJECT_SOURCE_DIR}/tests/test_onnx_export_contract.py
                        --validation-bundle ${TREESEM_TEST_BUNDLE_DIR}
                        --artifact ${TREESEM_TEST_ARTIFACT}
                        --raw-file ${TREESEM_TEST_RAW_FILE}
                )
                set_tests_properties(onnx_export_contract_test PROPERTIES
                    ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/PythonServices/TreeSemModelAdapter"
                    TIMEOUT 120
                )
            endif()
            add_test(
                NAME full_onnx_parity_test
                COMMAND ${TREESEM_MODEL_PYTHON_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/test_full_onnx_parity.py
                    --dump-executable $<TARGET_FILE:onnx_predictions_dump>
                    --bundle ${TREESEM_TEST_BUNDLE_DIR}
                    --raw-file ${TREESEM_TEST_RAW_FILE}
            )
            set_tests_properties(full_onnx_parity_test PROPERTIES
                ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/PythonServices/TreeSemModelAdapter"
                TIMEOUT 120
            )
            if(KAMA_BUILD_TREESEM_SERVER)
                add_test(
                    NAME onnx_backend_integration_test
                    COMMAND ${TREESEM_MODEL_PYTHON_EXECUTABLE}
                        ${PROJECT_SOURCE_DIR}/tests/test_onnx_backend_integration.py
                        $<TARGET_FILE:treesem_server>
                        ${TREESEM_TEST_BUNDLE_DIR}
                )
                set_tests_properties(onnx_backend_integration_test PROPERTIES
                    ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/PythonServices/TreeSemModelAdapter"
                    TIMEOUT 120
                )
                add_test(
                    NAME m4_business_integration_test
                    COMMAND ${Python3_EXECUTABLE}
                        ${PROJECT_SOURCE_DIR}/tests/test_m4_business_integration.py
                        $<TARGET_FILE:treesem_server>
                        ${TREESEM_TEST_BUNDLE_DIR}
                )
                set_tests_properties(m4_business_integration_test PROPERTIES
                    TIMEOUT 120)
            endif()
        endif()
    endif()

    if(KAMA_BUILD_TREESEM_SERVER)
        add_test(
            NAME m5_agent_integration_test
            COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_m5_agent_integration.py
                $<TARGET_FILE:treesem_server>
        )
        set_tests_properties(m5_agent_integration_test PROPERTIES TIMEOUT 30)
        add_test(
            NAME backend_async_integration_test
            COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_backend_integration.py
                $<TARGET_FILE:treesem_server>
        )
        if(KAMA_ENABLE_AUTH)
            add_test(
                NAME m6_auth_integration_test
                COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/test_m6_auth_integration.py
                    $<TARGET_FILE:treesem_server>
            )
            set_tests_properties(m6_auth_integration_test PROPERTIES TIMEOUT 30)
        endif()
        if(KAMA_ENABLE_MYSQL AND KAMA_ENABLE_ONNXRUNTIME
           AND TREESEM_ENABLE_MYSQL_INTEGRATION_TESTS
           AND TREESEM_TEST_BUNDLE_DIR)
            add_test(
                NAME m4_mysql_integration_test
                COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/test_m4_mysql_integration.py
                    $<TARGET_FILE:treesem_server>
                    ${TREESEM_TEST_BUNDLE_DIR}
            )
            set_tests_properties(m4_mysql_integration_test PROPERTIES TIMEOUT 120)
        endif()
    endif()
endif()
