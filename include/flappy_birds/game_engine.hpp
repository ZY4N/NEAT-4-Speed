#pragma once

#include "game_logic/config.hpp"
#include "game_logic/physics_engine.hpp"
#include "game_logic/state.hpp"
#include "rendering/color_config.hpp"
#include "rendering/color_renderer.hpp"
#include "rendering/texture_renderer.hpp"

#include <span>

namespace flappy_birds {

template<class Renderer>
class game_engine_t {
public:
	using renderer_config_t = rendering::texture_renderer_t::config_t;

	game_engine_t(
		const game_logic::config_t& game_config,
		const renderer_config_t& renderer_config,
		std::size_t bird_count,
		int window_width,
		int window_height
	);

	void flap(std::size_t index);

	bool update(float dt);

	void render(sf::RenderWindow& window);

	void reset();

	void default_view(int window_width, int window_height);

	game_logic::state_t& state();

	std::span<const float> scores();

public:
	game_logic::config_t m_game_config;
	rendering::view_config_t m_view_config;

private:
	std::size_t m_bird_count;
	game_logic::state_t m_game_state;
	game_logic::physics_engine_t m_physics_engine;
	Renderer m_renderer;

	std::vector<bool> m_will_flap;
};

} // namespace flappy_birds

#include "flappy_birds/game_engine.ipp"
