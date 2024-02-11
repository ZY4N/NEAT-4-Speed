#include "flappy_birds/rendering/color_renderer.hpp"

#include <filesystem>
#include <iostream>

namespace flappy_birds::rendering {

color_renderer_t::color_renderer_t() {
	const auto current_path = std::filesystem::current_path();
	const auto font_file = current_path / ".." / "assets" / "fonts" / "PixeloidSans-mLxMm.ttf";
	if (not score_font.loadFromFile(font_file)) {
		std::cerr << "Could not open font file: " << font_file << std::endl;
		exit(EXIT_FAILURE);
	}
	score_text.setFont(score_font);
	score_text.setFillColor(sf::Color::White);
	score_text.setOutlineColor(sf::Color::Black);
}

void color_renderer_t::render(
	const view_config_t& view_config,
	const color_config_t& color_config,
	const game_config_t& game_config,
	game_state_t& game_state,
	sf::RenderWindow& window
) {

	window.clear(color_config.background_color);

	const auto window_size = window.getSize();
	const auto window_width = static_cast<float>(window_size.x);
	const auto window_height = static_cast<float>(window_size.y);
	const auto half_window_width = window_width / 2.0f;
	const auto half_window_height = window_height / 2.0f;

	const auto to_window_space_x = [&](const float x) {
		return half_window_width + view_config.scale * (x - view_config.center_x);
	};
	const auto to_window_space_y = [&](const float y) {
		return half_window_height - view_config.scale * (y - view_config.center_y);
	};
	const auto measure_in_window_space = [&](const float size) { return view_config.scale * size; };

	// Draw ceiling
	const auto ceiling_window_y = to_window_space_y(game_config.ceiling_y);
	if (ceiling_window_y >= 0.0f) {
		ceiling_rect.setPosition(0, 0);
		ceiling_rect.setSize({ window_width, ceiling_window_y });
		ceiling_rect.setFillColor(color_config.ceiling_color);
		window.draw(ceiling_rect);
	}

	// Draw floor
	const auto floor_window_y = to_window_space_y(game_config.floor_y);
	if (ceiling_window_y <= window_height) {
		floor_rect.setPosition(0, floor_window_y);
		floor_rect.setSize({ window_width, window_height - ceiling_window_y });
		floor_rect.setFillColor(color_config.floor_color);
		window.draw(floor_rect);
	}

	// Draw pipes
	const auto pipe_window_width = measure_in_window_space(game_config.pipe_width);

	pipe_rect.setFillColor(color_config.pipe_color);

	for (std::size_t i{}; i != game_state.pipe_gaps_y.size(); ++i) {

		const auto pipe_position_x = game_state.pipe_position_x +
			game_config.pipe_spacing_x *
				static_cast<float>(static_cast<int>(i) - static_cast<int>(game_config.pipes_behind_bird));

		const auto pipe_window_pos_x = to_window_space_x(pipe_position_x);

		const auto pipe_gap_y = game_state.pipe_gaps_y[i];
		const auto window_pipe_upper_edge_y = to_window_space_y(pipe_gap_y + game_config.pipe_spacing_y / 2.0f);
		const auto window_pipe_lower_edge_y = to_window_space_y(pipe_gap_y - game_config.pipe_spacing_y / 2.0f);

		if (window_pipe_upper_edge_y > 0.0f) {
			pipe_rect.setPosition(pipe_window_pos_x, ceiling_window_y);
			pipe_rect.setSize({ pipe_window_width, window_pipe_upper_edge_y - ceiling_window_y });
			window.draw(pipe_rect);
		}

		if (window_pipe_lower_edge_y < window_height) {
			pipe_rect.setPosition(pipe_window_pos_x, window_pipe_lower_edge_y);
			pipe_rect.setSize({ pipe_window_width, floor_window_y - window_pipe_lower_edge_y });
			window.draw(pipe_rect);
		}
	}

	const auto window_bird_radius = measure_in_window_space(game_config.bird_radius);
	bird_circ.setFillColor(color_config.bird_color);
	bird_circ.setRadius(window_bird_radius);

	const auto window_bird_pos_x = to_window_space_x(game_config.bird_x);
	for (std::size_t i{}; i != game_state.bird_states.size(); ++i) {
		const auto window_bird_pos_y = to_window_space_y(game_state.bird_states[i].position_y);
		if (game_config.bird_radius <= window_bird_pos_y and
		    window_bird_pos_y + game_config.bird_radius <= window_height) {
			bird_circ.setPosition(window_bird_pos_x - window_bird_radius, window_bird_pos_y - window_bird_radius);
			window.draw(bird_circ);
		}
	}

	const auto score_str = std::to_string(game_state.pipes_surpassed_count);
	score_text.setString(score_str);

	const auto text_size = 0.2f * window_height;
	score_text.setCharacterSize(static_cast<unsigned int>(text_size));
	score_text.setOutlineThickness(0.05f * text_size);

	const auto text_dim = score_text.getGlobalBounds().getSize();
	const auto text_pos = sf::Vector2f(half_window_width - text_dim.x / 2.0f, window_height * 0.1f - text_dim.y / 2.0f);

	score_text.setPosition(text_pos);
	window.draw(score_text);
}

} // namespace flappy_birds::rendering
