#include "neat/helpers/connection_lookup.hpp"

#include "neat/types.hpp"

#include <optional>

namespace neat {

connection_lookup::sorted_node_pair::sorted_node_pair(
	const types::node_index_t& node_a, const types::node_index_t& node_b
) :
	smaller{ node_a }, bigger{ node_b } {
	if (bigger < smaller) {
		std::swap(bigger, smaller);
	}
}

void connection_lookup::clear() {
	m_node_pairs.clear();
}

std::pair<std::size_t, bool> connection_lookup::lookup_connection(const sorted_node_pair& conn) {
	const auto entry_it = std::lower_bound(m_node_pairs.begin(), m_node_pairs.end(), conn);
	auto index = entry_it - m_node_pairs.begin();
	auto found = entry_it != m_node_pairs.begin() and *(entry_it - 1) == conn;

	return { index, found };
}

types::innovation_number_t connection_lookup::update_connection_info(
	debug_span<types::connection_info_t> innovation_numbers,
	const types::conn_index_t& conn_index,
	const types::node_index_t& from,
	const types::node_index_t& to
) {

	const auto node_pair = sorted_node_pair(from, to);

	while (lock.test_and_set(std::memory_order_acquire)) {
		// acquire lock
	}

	auto [index, found] = lookup_connection(node_pair);

	types::innovation_number_t inno_num;
	if (found) {
		inno_num = innovation_numbers[index].innovation_number;
	} else {
		inno_num = m_innovation_counter++;
		m_node_pairs.insert(m_node_pairs.begin() + index, node_pair);
	}

	lock.clear(std::memory_order_release);

	innovation_numbers[conn_index].enabled = true;
	innovation_numbers[conn_index].innovation_number = inno_num;

	return inno_num;
}

} // namespace neat
