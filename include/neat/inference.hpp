#pragma once

#include "types.hpp"

namespace neat::inference {

namespace types {

using node_index_t = std::size_t;
using conn_index_t = std::size_t;

using node_index_range_t = integer_range<node_index_t>;
using conn_index_range_t = integer_range<conn_index_t>;

using value_t = float;

struct connection_t {
	node_index_t source_node_index;
	neat::types::connection_weight_t weight;
};

struct node_t {
	node_index_t node_index;
	conn_index_t incoming_connection_count;
};

struct network_t {
	node_index_range_t nodes;
	conn_index_t connections_begin;
};

struct network_group_t {
	debug_vector<network_t> networks;
	debug_vector<node_t> nodes;
	debug_vector<connection_t> connections;
};

} // namespace types

void evaluate_network_range(
	const types::network_group_t& network_group,
	debug_span<const types::value_t> inputs,
	debug_span<types::value_t> network_outputs,
	const neat::types::network_range_t& network_range
);

types::value_t activation_function(const types::value_t& signal);

} // namespace neat::inference
