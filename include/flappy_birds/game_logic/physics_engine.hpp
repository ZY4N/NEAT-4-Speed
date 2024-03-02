#pragma once

#include "config.hpp"
#include "state.hpp"

#include <random>
#include "util/debug_vector.hpp" // TODO remove

namespace flappy_birds::game_logic {

class physics_engine_t {
public:
	bool update(const config_t& config, state_t& state, debug_vector<bool>& flap, float dt);

private:
	debug_vector<bool> collided;
};

} // namespace flappy_birds::game_logic
