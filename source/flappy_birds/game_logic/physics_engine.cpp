#include "flappy_birds/game_logic/physics_engine.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace flappy_birds::game_logic {

bool physics_engine_t::update(const config_t& config, state_t& state, std::vector<bool>& flap, float dt) {

	const auto bird_radius_sq = std::pow(config.bird_radius, 2);

	static std::uniform_real_distribution gap_distrib_y(
		config.floor_y + config.pipe_spacing_y / 2.0f,
		config.ceiling_y - config.pipe_spacing_y / 2.0f
	);

	// Update pipes
	state.pipe_position_x += config.pipe_velocity_x * dt;
	if (state.pipe_position_x + config.pipe_width < config.bird_x - config.bird_radius) {

		state.pipe_gaps_y.pop_front();
		state.pipe_gaps_y.push_back(gap_distrib_y(state.rng));

		state.pipe_position_x += config.pipe_spacing_x;
		++state.pipes_surpassed_count;
	}

	// Update physics for each bird
	for (std::size_t i{}; i != state.bird_states.size(); ++i) {
		auto& bird_state = state.bird_states[i];
		const auto flap_scale = static_cast<float>(flap[i]);
		bird_state.velocity_y = std::lerp(
			bird_state.velocity_y + config.gravitational_acceleration_y * dt,
			config.bird_flap_velocity_y,
			flap_scale
		);
		bird_state.position_y += bird_state.velocity_y * dt;
		bird_state.seconds_since_flap = (1.0f - flap_scale) * (bird_state.seconds_since_flap + dt);
		flap[i] = false;
	}

	const auto next_pipe_gap_y = state.pipe_gaps_y[config.pipes_behind_bird];
	const auto upper_pipe_edge_y = next_pipe_gap_y + config.pipe_spacing_y / 2.0f;
	const auto lower_pipe_edge_y = next_pipe_gap_y - config.pipe_spacing_y / 2.0f;

	float closest_point_in_pipe_x = std::clamp(
		config.bird_x,
		state.pipe_position_x,
		state.pipe_position_x + config.pipe_width
	);

	collided.clear();
	collided.resize(state.bird_states.size(), false);

	for (std::size_t i{}; i != state.bird_states.size(); ++i) {
		auto& bird_state = state.bird_states[i];

		// ceiling
		if (bird_state.position_y + config.bird_radius > config.ceiling_y) {
			collided[i] = true;
			continue;
		}

		// floor
		if (bird_state.position_y - config.bird_radius < config.floor_y) {
			collided[i] = true;
			continue;
		}

		{
			// upper pipe
			const auto closest_point_in_upper_pipe_y = std::max(bird_state.position_y, upper_pipe_edge_y);
			const auto dist_to_upper_pipe_sq =
				(std::pow(closest_point_in_pipe_x - config.bird_x, 2) +
			     std::pow(closest_point_in_upper_pipe_y - bird_state.position_y, 2));

			if (dist_to_upper_pipe_sq <= bird_radius_sq) {
				collided[i] = true;
				continue;
			}
		}

		{
			// lower pipe
			const auto closest_point_in_lower_pipe_y = std::min(bird_state.position_y, lower_pipe_edge_y);
			const auto dist_to_lower_pipe_sq =
				(std::pow(closest_point_in_pipe_x - config.bird_x, 2) +
			     std::pow(closest_point_in_lower_pipe_y - bird_state.position_y, 2));

			if (dist_to_lower_pipe_sq <= bird_radius_sq) {
				collided[i] = true;
				continue;
			}
		}
	}

	auto bird_state_it = state.bird_states.begin();
	auto bird_indices_it = state.active_bird_indices.begin();

	const auto distance_from_start = config.pipe_spacing_x * static_cast<float>(state.pipes_surpassed_count) +
		(config.pipe_spacing_x - state.pipe_position_x);

	const auto distance_from_start_score = config.score_weight_dist_from_start_x * distance_from_start;
	const auto max_dist_from_gap_y = gap_distrib_y.max() - gap_distrib_y.min();

	for (std::size_t i{}; i != collided.size(); ++i) {
		if (collided[i]) {
			const auto original_index = state.active_bird_indices[i];

			const auto& bird_y = state.bird_states[i].position_y;

			const auto closes_pipe_opening_y = std::clamp(
				bird_y,
				lower_pipe_edge_y + config.bird_radius,
				upper_pipe_edge_y - config.bird_radius
			);

			const auto dist_from_gap_y = std::abs(closes_pipe_opening_y - bird_y);

			state.scores[original_index] += distance_from_start_score +
				config.score_weight_dist_from_gap_y * (1.0f - dist_from_gap_y / max_dist_from_gap_y);

		} else {
			*bird_state_it++ = state.bird_states[i];
			*bird_indices_it++ = state.active_bird_indices[i];
		}
	}

	state.bird_states.resize(bird_state_it - state.bird_states.begin());
	state.active_bird_indices.resize(bird_indices_it - state.active_bird_indices.begin());

	return state.bird_states.empty();
}

} // namespace flappy_birds::game_logic
