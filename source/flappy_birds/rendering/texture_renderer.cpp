#include "flappy_birds/rendering/texture_renderer.hpp"

#include <filesystem>
#include <functional>
#include <iostream>
#include <numbers>

namespace flappy_birds::rendering {

texture_renderer_t::texture_renderer_t(const texture_config_t& texture_config) : m_texture_config{ texture_config } {
	const auto current_path = std::filesystem::current_path();
	const auto assets_dir = current_path / ".." / "assets";
	const auto font_dir = assets_dir / "fonts";
	const auto texture_dir = assets_dir / "textures";

	const auto font_file = font_dir / m_texture_config.score_font;
	if (not score_font.loadFromFile(font_file)) {
		std::cerr << "Could not open font file: " << font_file << std::endl;
		exit(EXIT_FAILURE);
	}

	const auto bird_texture_file = texture_dir / m_texture_config.bird_texture;
	bird_texture.loadFromFile(bird_texture_file.c_str());
	bird_sprite.setTexture(bird_texture, true);

	const auto pipe_texture_file = texture_dir / m_texture_config.pipe_texture;
	pipe_texture.loadFromFile(pipe_texture_file.c_str());
	pipe_sprite.setTexture(pipe_texture, true);

	score_text.setFont(score_font);
	score_text.setFillColor(sf::Color::White);
	score_text.setOutlineColor(sf::Color::Black);
}

[[maybe_unused]] static sf::Color color_hash(std::uint32_t index) {
	static constexpr std::uint32_t c2 = 0x27d4'eb2d; // a prime or an odd constant

	index = (index ^ 61) ^ (index >> 16);
	index = index + (index << 3);
	index = index ^ (index >> 4);
	index = index * c2;
	index = index ^ (index >> 15);

	const auto region = (index >> 8) << 8;
	const auto x = index & 0xFF;

	std::uint8_t r{}, g{}, b{};
	switch (region % 6) {
	case 0:
		r = 255;
		g = 0;
		b = 0;
		g += x;
		break;
	case 1:
		r = 255;
		g = 255;
		b = 0;
		r -= x;
		break;
	case 2:
		r = 0;
		g = 255;
		b = 0;
		b += x;
		break;
	case 3:
		r = 0;
		g = 255;
		b = 255;
		g -= x;
		break;
	case 4:
		r = 0;
		g = 0;
		b = 255;
		r += x;
		break;
	case 5:
		r = 255;
		g = 0;
		b = 255;
		b -= x;
		break;
	}
	return sf::Color(r, g, b);
}

void texture_renderer_t::render(
	const view_config_t& view_config,
	const game_config_t& game_config,
	game_state_t& game_state,
	sf::RenderWindow& window
) {

	window.clear(m_texture_config.background_color);

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

	const auto ceiling_window_y = to_window_space_y(game_config.ceiling_y);
	const auto floor_window_y = to_window_space_y(game_config.floor_y);

	// Draw pipes
	const auto pipe_window_width = measure_in_window_space(game_config.pipe_width);

	pipe_rect.setFillColor(m_texture_config.pipe_color);

	const auto [pipe_width, pipe_height] = pipe_texture.getSize();
	const auto pipe_scale = 3.0f * pipe_window_width / static_cast<float>(pipe_width); // 2/3 Pipe image has padding
	pipe_sprite.setScale(pipe_scale, pipe_scale);

	for (std::size_t i{}; i != game_state.pipe_gaps_y.size(); ++i) {

		const auto pipe_position_x = game_state.pipe_position_x +
			game_config.pipe_spacing_x *
				static_cast<float>(static_cast<int>(i) - static_cast<int>(game_config.pipes_behind_bird));

		const auto pipe_window_pos_x = to_window_space_x(pipe_position_x);

		const auto pipe_gap_y = game_state.pipe_gaps_y[i];
		const auto window_pipe_upper_edge_y = to_window_space_y(pipe_gap_y + game_config.pipe_spacing_y / 2.0f);
		const auto window_pipe_lower_edge_y = to_window_space_y(pipe_gap_y - game_config.pipe_spacing_y / 2.0f);

		if (window_pipe_upper_edge_y > 0.0f) {

			pipe_sprite.setRotation(180.0f);
			pipe_sprite.setScale(-pipe_scale, pipe_scale);
			pipe_sprite.setPosition(pipe_window_pos_x - pipe_window_width, window_pipe_upper_edge_y);
			window.draw(pipe_sprite);

			/*pipe_rect.setPosition(pipe_window_pos_x, ceiling_window_y);
			pipe_rect.setSize({ pipe_window_width, window_pipe_upper_edge_y - ceiling_window_y });
			window.draw(pipe_rect);*/
		}

		if (window_pipe_lower_edge_y < window_height) {

			pipe_sprite.setRotation(0.0f);
			pipe_sprite.setScale(pipe_scale, pipe_scale);
			pipe_sprite.setPosition(pipe_window_pos_x - pipe_window_width, window_pipe_lower_edge_y);
			window.draw(pipe_sprite);

			/*pipe_rect.setPosition(pipe_window_pos_x, window_pipe_lower_edge_y);
			pipe_rect.setSize({ pipe_window_width, floor_window_y - window_pipe_lower_edge_y });
			window.draw(pipe_rect);*/
		}
	}

	// Draw ceiling
	if (ceiling_window_y >= 0.0f) {
		ceiling_rect.setPosition(0, 0);
		ceiling_rect.setSize({ window_width, ceiling_window_y });
		ceiling_rect.setFillColor(m_texture_config.ceiling_color);
		window.draw(ceiling_rect);
	}

	// Draw floor
	if (ceiling_window_y <= window_height) {
		floor_rect.setPosition(0, floor_window_y);
		floor_rect.setSize({ window_width, window_height - ceiling_window_y });
		floor_rect.setFillColor(m_texture_config.floor_color);
		window.draw(floor_rect);
	}

	const auto window_bird_radius = measure_in_window_space(game_config.bird_radius);
	// bird_circ.setFillColor(m_texture_config.bird_color);
	// bird_circ.setRadius(window_bird_radius);

	const auto [bird_sheet_width, bird_height] = bird_texture.getSize();
	const auto bird_width = bird_sheet_width / m_texture_config.bird_animation_frame_count;

	const auto bird_scale = 2.0f *
		std::min(window_bird_radius / static_cast<float>(bird_width),
	             window_bird_radius / static_cast<float>(bird_height));

	bird_sprite.setOrigin(bird_width / 2, bird_height / 2);
	bird_sprite.setScale(bird_scale, bird_scale);

	const auto window_bird_pos_x = to_window_space_x(game_config.bird_x);
	for (std::size_t i{}; i != game_state.bird_states.size(); ++i) {
		const auto& bird_state = game_state.bird_states[i];

		const auto window_bird_pos_y = to_window_space_y(bird_state.position_y);
		// if (game_config.bird_radius <= window_bird_pos_y and
		//     window_bird_pos_y + game_config.bird_radius <= window_height) {

		// bird_circ.setPosition(window_bird_pos_x - window_bird_radius, window_bird_pos_y - window_bird_radius);
		// window.draw(bird_circ);

		auto animation_frame_index = 1 + // Flap-frame should be second animation frame
			static_cast<int>(bird_state.seconds_since_flap / m_texture_config.bird_animation_frame_duration);

		if (animation_frame_index >= m_texture_config.bird_animation_frame_count) {
			animation_frame_index = 0;
		}

		bird_sprite.setTextureRect({ animation_frame_index * static_cast<int>(bird_width),
		                             0,
		                             static_cast<int>(bird_width),
		                             static_cast<int>(bird_height) });
		bird_sprite.setPosition(window_bird_pos_x, window_bird_pos_y);
		bird_sprite.setRotation(
			-0.4f * std::atan2(bird_state.velocity_y, -game_config.pipe_velocity_x) *
			(180.f / std::numbers::pi_v<float>)
		);

		bird_sprite.setColor(color_hash(game_state.active_bird_indices[i]));

		window.draw(bird_sprite);
		//}
	}

	const auto score_str = std::to_string(game_state.pipes_surpassed_count);
	score_text.setString(score_str);

	const auto text_size = 0.15f * window_height;
	score_text.setCharacterSize(static_cast<unsigned int>(text_size));
	score_text.setOutlineThickness(0.05f * text_size);

	const auto text_dim = score_text.getGlobalBounds().getSize();
	const auto text_pos = sf::Vector2f(half_window_width - text_dim.x / 2.0f, window_height * 0.1f - text_dim.y / 2.0f);

	score_text.setPosition(text_pos);
	window.draw(score_text);
}

} // namespace flappy_birds::rendering
