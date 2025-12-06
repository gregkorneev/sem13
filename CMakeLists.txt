cmake_minimum_required(VERSION 3.10)
project(sem13 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Создаем исполняемый файл
add_executable(sem13
    src/main.cpp
    src/dataset.cpp
    src/id3.cpp
    src/tree_utils.cpp
)

# Теперь, когда sem13 создан — подключаем include/
target_include_directories(sem13 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
