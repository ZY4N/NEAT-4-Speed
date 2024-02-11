#pragma once

#include "neat/evolution_config.hpp"
#include "neat/types.hpp"

#include <atomic>
#include <deque>

namespace neat {

class species_sorter {
private:
	struct species_bucket_t {
		std::atomic_flag lock = ATOMIC_FLAG_INIT;
		debug_vector<types::network_t> networks;
	};

public:
	void clear();

	void sort_into_buckets(
		const difference_config_t& config,
		debug_span<const types::connection_weight_t> connection_weights,
		debug_span<const types::connection_info_t> connection_infos,
		debug_span<const types::network_t> networks,
		const types::network_index_t& approx_bucket_size,
		const types::network_range_t& network_range
	);

	void assign_species_and_sorted_networks(
		debug_vector<types::species_t>& all_species, debug_span<types::network_t> networks
	);

protected:
	types::species_index_t search_matching_network(
		const difference_config_t& config,
		debug_span<const types::connection_weight_t> connection_weights,
		debug_span<const types::connection_info_t> connection_infos,
		const types::network_t& network,
		const types::network_range_t& lookup_range
	);

	static float network_difference(
		const difference_config_t& config,
		debug_span<const types::connection_weight_t> connection_weights,
		debug_span<const types::connection_info_t> connection_infos,
		const types::network_t& network_a,
		const types::network_t& network_b
	) ;

private:
	std::atomic_flag m_lookup_lock = ATOMIC_FLAG_INIT;
	std::deque<types::network_t> m_characteristic_networks;
	std::deque<species_bucket_t> m_species_buckets;
};

} // namespace neat
