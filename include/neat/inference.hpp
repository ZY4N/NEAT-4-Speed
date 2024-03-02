#pragma once

#include "types.hpp"

namespace neat::inference {

namespace types {

using node_index_t = std::uint32_t;
using abs_conn_index_t = std::size_t;
using rel_conn_index_t = std::uint32_t;

using abs_conn_index_range_t = integer_range<abs_conn_index_t>;
using rel_conn_index_range_t = integer_range<rel_conn_index_t>;

using value_t = float;

struct weighted_connection_t {
	static_assert(sizeof(node_index_t) == sizeof(value_t), "Sizes must be equal for efficient alignment.");
	node_index_t source_node_index;
	neat::types::connection_weight_t weight;
};

struct network_t {
	abs_conn_index_t incoming_connections_begin;
	abs_conn_index_range_t incoming_connection_count_range;
};

struct network_group_t {
	debug_vector<network_t> networks;
	// After each networks incoming_connection_counts this also stores the output node lookup.
	// | node 0  | node 1  | node 2  | output map |
	// | 4 conns | 5 conns | 2 conns |  2  |   1  |
	static_assert(
		std::numeric_limits<node_index_t>::max() <= std::numeric_limits<rel_conn_index_t>::max(),
		"A connection index needs to be able to represent a node index (because of output node map)."
	);
	debug_vector<rel_conn_index_t> incoming_connection_counts_and_node_lookups;
	debug_vector<weighted_connection_t> connections;
};

} // namespace types

inline constexpr auto invalid_node_index = std::numeric_limits<types::node_index_t>::max();

void evaluate_network_range(
	const types::network_group_t& network_group,
	debug_span<const types::value_t> inputs,
	debug_span<types::value_t> network_outputs,
	const neat::types::network_range_t& network_range
);

types::value_t activation_function(const types::value_t& signal);

} // namespace neat::inference
