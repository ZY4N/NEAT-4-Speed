#pragma once

#include <SFML/Graphics/Color.hpp>

namespace flappy_birds::rendering {

struct color_config_t {
	sf::Color background_color{ 135, 206, 250 };
	sf::Color ceiling_color{ 0, 150, 255 };
	sf::Color floor_color{ 194, 178, 128 };
	sf::Color bird_color{ 255, 36, 0 };
	sf::Color pipe_color{ 50, 205, 50 };
};

} // namespace flappy_birds::rendering
