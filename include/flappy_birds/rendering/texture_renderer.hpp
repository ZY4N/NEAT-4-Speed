#pragma once

#include "flappy_birds/game_logic/config.hpp"
#include "flappy_birds/game_logic/state.hpp"
#include "flappy_birds/rendering/texture_config.hpp"
#include "flappy_birds/rendering/view_config.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Graphics/Texture.hpp>
#include "util/debug_vector.hpp" // TODO remove

namespace flappy_birds::rendering {

using game_config_t = game_logic::config_t;
using game_state_t = game_logic::state_t;

class texture_renderer_t {
public:
	using config_t = texture_config_t;

	texture_renderer_t(const texture_config_t& texture_config);

	void render(
		const view_config_t& view_config,
		const game_config_t& game_config,
		game_state_t& game_state,
		sf::RenderWindow& window
	);

private:
	const texture_config_t m_texture_config;

	sf::Font score_font;
	sf::Text score_text;
	sf::CircleShape bird_circ;
	sf::RectangleShape pipe_rect;
	sf::RectangleShape floor_rect, ceiling_rect;

	sf::Texture bird_texture;
	sf::Sprite bird_sprite;

	sf::Texture pipe_texture;
	sf::Sprite pipe_sprite;
};

} // namespace flappy_birds::rendering
