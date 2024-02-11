#pragma once

#include "neat/types.hpp"

#include <cinttypes>
#include <span>
#include "util/debug_span.hpp" // TODO remove
#include <vector>
#include "util/debug_vector.hpp" // TODO remove
#include <atomic>

namespace neat {

class connection_lookup {
private:
	struct sorted_node_pair {
		types::node_index_t smaller, bigger;

		sorted_node_pair(const types::node_index_t& node_a, const types::node_index_t& node_b);
		friend auto operator<=>(const sorted_node_pair&, const sorted_node_pair&) = default;
	};

public:
	void clear();

	types::innovation_number_t update_connection_info(
		debug_span<types::connection_info_t> innovation_numbers,
		const types::conn_index_t& conn_index,
		const types::node_index_t& from,
		const types::node_index_t& to
	);

private:
	std::pair<std::size_t, bool> lookup_connection(const sorted_node_pair& conn);

	std::atomic_flag lock = ATOMIC_FLAG_INIT;
	types::innovation_number_t m_innovation_counter;
	debug_vector<sorted_node_pair> m_node_pairs;
};

} // namespace neat
