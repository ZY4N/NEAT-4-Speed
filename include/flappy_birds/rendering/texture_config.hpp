#pragma once

#include <SFML/Graphics/Color.hpp>

namespace flappy_birds::rendering {

struct texture_config_t {
	sf::Color background_color{ 135, 206, 250 };
	sf::Color ceiling_color{ 0, 150, 255 };
	sf::Color floor_color{ 194, 178, 128 };
	sf::Color bird_color{ 255, 36, 0 };
	sf::Color pipe_color{ 50, 205, 50 };
	const int bird_animation_frame_count{ 4 };
	float bird_animation_frame_duration{ 0.1f };
	const std::string_view score_font{ "PixeloidSans-mLxMm.ttf" };
	const std::string_view bird_texture{ "birdsheet.png" };
	const std::string_view pipe_texture{ "pipe.png" };
};

} // namespace flappy_birds::rendering
