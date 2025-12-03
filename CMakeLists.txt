cmake_minimum_required(VERSION 3.16)

project(sem13_id3 LANGUAGES CXX)

# Стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Исполняемый файл
add_executable(sem13
    src/main.cpp
)

# Подключаем директорию с заголовками (на будущее)
target_include_directories(sem13
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
