#pragma once

#include <cinttypes>

namespace flappy_birds::game_logic {

struct config_t {
	float gravitational_acceleration_y{ -9.8 };

	float bird_flap_velocity_y{ 2.4 };
	float bird_x{ 0.0 };
	float bird_radius{ 0.1 };

	float pipe_velocity_x{ -0.9 };
	float pipe_width{ 0.3 };
	float pipe_spacing_x{ 1.5 };
	float pipe_spacing_y{ 0.7 };

	float floor_y{ 0.0 };
	float ceiling_y{ 2.0 };

	float score_weight_dist_from_start_x{ 1.0 };
	float score_weight_dist_from_gap_y{ 0.1 };

	std::uint32_t pipes_in_front_of_bird{ 2 };
	std::uint32_t pipes_behind_bird{ 1 };
};

} // namespace flappy_birds::game_logic
