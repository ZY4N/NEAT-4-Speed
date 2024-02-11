cmake_minimum_required(VERSION 3.28)
project(NEAT-4-Speed)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -pedantic -Wall -Werror ")

add_executable(NEAT-4-Speed main.cpp
        include/flappy_birds/game_engine.hpp
        include/flappy_birds/game_logic/config.hpp
        include/flappy_birds/game_logic/physics_engine.hpp
        include/flappy_birds/game_logic/state.hpp
        include/flappy_birds/rendering/color_config.hpp
        include/flappy_birds/rendering/color_renderer.hpp
        include/flappy_birds/rendering/texture_config.hpp
        include/flappy_birds/rendering/texture_renderer.hpp
        include/flappy_birds/rendering/view_config.hpp
        include/neat/evolution_config.hpp
        include/neat/helpers/connection_lookup.hpp
        include/neat/helpers/species_sorter.hpp
        include/neat/inference.hpp
        include/neat/network_interface_config.hpp
        include/neat/trainer.hpp
        include/neat/trainer.hpp
        include/neat/types.hpp
        include/util/debug_span.hpp
        include/util/debug_vector.hpp
        include/util/integer_range.hpp
        source/flappy_birds/game_engine.ipp
        source/flappy_birds/game_logic/physics_engine.cpp
        source/flappy_birds/rendering/color_renderer.cpp
        source/flappy_birds/rendering/texture_renderer.cpp
        source/neat/helpers/connection_lookup.cpp
        source/neat/helpers/species_sorter.cpp
        source/neat/inference.cpp
        source/neat/trainer.cpp
)

include_directories(
        include
        source
)

find_package(SFML REQUIRED COMPONENTS graphics system)

include_directories(${SFML_INCLUDE_DIR})
target_link_libraries(NEAT-4-Speed sfml-graphics sfml-system sfml-window ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
