#pragma once

#include "types.hpp"

namespace neat {

struct network_interface_config_t {
	types::node_index_t input_count, output_count;
};

} // namespace neat
