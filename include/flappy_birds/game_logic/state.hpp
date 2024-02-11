#pragma once

#include <cinttypes>
#include <deque>
#include <random>
#include <vector>

namespace flappy_birds::game_logic {

struct bird_state_t {
	float position_y;
	float velocity_y;
	float seconds_since_flap; // This is only for rendering, but I don't care rn.
};

struct state_t {
	std::vector<bird_state_t> bird_states;
	std::vector<std::uint32_t> active_bird_indices;
	std::vector<float> scores;
	std::deque<float> pipe_gaps_y;
	float pipe_position_x;
	std::size_t pipes_surpassed_count;
	std::default_random_engine rng{ std::random_device{}() };
};

} // namespace flappy_birds::game_logic
