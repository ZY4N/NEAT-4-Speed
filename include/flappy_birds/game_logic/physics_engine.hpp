#pragma once

#include "config.hpp"
#include "state.hpp"

#include <random>
#include <vector>

namespace flappy_birds::game_logic {

class physics_engine_t {
public:
	bool update(const config_t& config, state_t& state, std::vector<bool>& flap, float dt);

private:
	std::vector<bool> collided;
};

} // namespace flappy_birds::game_logic
