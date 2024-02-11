#pragma once

#include "flappy_birds/game_logic/config.hpp"
#include "flappy_birds/game_logic/state.hpp"
#include "flappy_birds/rendering/color_config.hpp"
#include "flappy_birds/rendering/view_config.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Text.hpp>
#include <vector>

namespace flappy_birds::rendering {

using game_config_t = game_logic::config_t;
using game_state_t = game_logic::state_t;

class color_renderer_t {
public:
	using config_t = color_config_t;

	color_renderer_t();

	void render(
		const view_config_t& view_config,
		const color_config_t& color_config,
		const game_config_t& game_config,
		game_state_t& game_state,
		sf::RenderWindow& window
	);

private:
	sf::Font score_font;
	sf::Text score_text;
	sf::CircleShape bird_circ;
	sf::RectangleShape pipe_rect;
	sf::RectangleShape floor_rect, ceiling_rect;
};

} // namespace flappy_birds::rendering
