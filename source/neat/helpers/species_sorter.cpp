#include "neat/helpers/species_sorter.hpp"

#include <algorithm>
#include <iostream> // TODO remove

namespace neat {

void species_sorter::clear() {
	m_characteristic_networks.clear();
	m_species_buckets.clear();
}

void species_sorter::sort_into_buckets(
	const difference_config_t& config,
	debug_span<const types::connection_weight_t> connection_weights,
	debug_span<const types::connection_info_t> connection_infos,
	debug_span<const types::network_t> networks,
	const types::network_index_t& approx_bucket_size,
	const types::network_range_t& network_range
) {
	for (const auto& network : network_range.span(networks)) {

		while (m_lookup_lock.test_and_set(std::memory_order_acquire)) {
			// acquire lookup lock
		}
		const auto known_species = types::species_range_t::from_index_count(0, m_characteristic_networks.size());
		m_lookup_lock.clear(std::memory_order_release);

		auto matching_index = search_matching_network(
			config,
			connection_weights,
			connection_infos,
			network,
			known_species
		);

		// TODO Some kind of deadlock I guess

		if (matching_index == invalid_species_index) {
			while (m_lookup_lock.test_and_set(std::memory_order_acquire)) {
				// re-acquire lookup lock
			}
			const auto unchecked_entries = types::species_range_t::from_begin_end(
				known_species.end(),
				m_characteristic_networks.size()
			);
			matching_index = search_matching_network(
				config,
				connection_weights,
				connection_infos,
				network,
				unchecked_entries
			);
			if (matching_index == invalid_species_index) {
				matching_index = m_characteristic_networks.size();
				m_characteristic_networks.push_back(network);
				auto& new_bucket = m_species_buckets.emplace_back();
				new_bucket.networks.reserve(approx_bucket_size);
			}
			m_lookup_lock.clear(std::memory_order_release);
		}

		auto& bucket = m_species_buckets[matching_index];
		while (bucket.lock.test_and_set(std::memory_order_acquire)) {
			// acquire bucket lock
		}
		bucket.networks.push_back(network);
		bucket.lock.clear(std::memory_order_release);
	}
}

void species_sorter::assign_species_and_sorted_networks(
	debug_vector<types::species_t>& all_species, debug_span<types::network_t> networks
) {
	all_species.resize(m_characteristic_networks.size());
	std::cout << "Differentiated " << all_species.size() << " species\n";

	auto network_index = types::species_index_t{};

	for (types::species_index_t i = 0; i != m_species_buckets.size(); ++i) {
		const auto& species_bucket = m_species_buckets[i];

		std::copy(species_bucket.networks.begin(), species_bucket.networks.end(), networks.begin() + network_index);

		auto& species = all_species[i];
		species.networks = types::species_range_t::from_index_count(network_index, species_bucket.networks.size());

		network_index = species.networks.end();
	}
}

types::species_index_t species_sorter::search_matching_network(
	const difference_config_t& config,
	debug_span<const types::connection_weight_t> connection_weights,
	debug_span<const types::connection_info_t> connection_infos,
	const types::network_t& network,
	const types::network_range_t& lookup_range
) {
	for (const auto& entry_index : lookup_range.indices()) {
		auto& characteristic_network = m_characteristic_networks[entry_index];
		if (network_difference(config, connection_weights, connection_infos, network, characteristic_network) <
		    config.difference_threshold) {
			return entry_index;
		}
	}
	return invalid_species_index;
}

float species_sorter::network_difference(
	const difference_config_t& config,
	debug_span<const types::connection_weight_t> connection_weights,
	debug_span<const types::connection_info_t> connection_infos,
	const types::network_t& network_a,
	const types::network_t& network_b
) {
	static constexpr auto num_parents = std::size_t{ 2 };

	auto network_ptrs = std::array{ &network_a, &network_b };

	std::array<types::conn_range_t, num_parents> network_connections;
	std::transform(
		network_ptrs.begin(),
		network_ptrs.end(),
		network_connections.begin(),
		[&](const types::network_t* ancestor_network) { return ancestor_network->connections; }
	);

	auto fit_index = std::size_t{ 0 }, unfit_index = std::size_t{ 1 };


	types::conn_index_t num_matching{}, num_disjoint{}, num_excess{};
	float total_matching_weight_delta{};

	// Get max size, before miss-using ranges as iterator indices.
	auto max_connections = std::max(network_connections[fit_index].size(), network_connections[unfit_index].size());

	while (std::none_of(network_connections.begin(), network_connections.end(), [](const auto& parent_connection) {
		return parent_connection.begin() == parent_connection.end();
	})) {
		auto& fit_conn_index = network_connections[fit_index].begin();
		auto& unfit_conn_index = network_connections[unfit_index].begin();

		const auto& fit_inno_num = connection_infos[fit_conn_index].innovation_number;
		const auto& unfit_inno_num = connection_infos[unfit_conn_index].innovation_number;

		if (fit_inno_num == unfit_inno_num) { // matching gene
			total_matching_weight_delta += std::abs(
				connection_weights[fit_conn_index] - connection_weights[unfit_conn_index]
			);
			++num_matching;
			++fit_conn_index;
			++unfit_conn_index;
		} else {
			const auto fit_connection_next = fit_inno_num < unfit_inno_num;
			++num_disjoint;
			if (fit_connection_next) { // fit disjoint gene
				++fit_conn_index;
			} else { // unfit disjoint gene
				++unfit_conn_index;
			}
		}
	}

	// Handler excess genes
	const auto excess_range_it = std::find_if(
		network_connections.begin(),
		network_connections.end(),
		[](const auto& parent_connection) { return parent_connection.begin() != parent_connection.end(); }
	);
	if (excess_range_it != network_connections.end()) {
		num_excess = excess_range_it->size();
	}

	// "N can be set to 1 if both genomes are small, i.e., consist of fewer than 20 genes" ¯\_(ツ)_/¯
	if (max_connections < 20) {
		max_connections = 1;
	}
	const auto normalization_scale = 1.0f / static_cast<float>(max_connections);

	const auto delta =
		(config.difference_disjoint_weight * static_cast<float>(num_disjoint) * normalization_scale +
	     config.difference_excess_weight * static_cast<float>(num_excess) * normalization_scale +
	     (num_matching == 0 ? 0
	                        : config.difference_avg_weight_weights *
	              (total_matching_weight_delta / static_cast<float>(num_matching))));

	return delta;
}

} // namespace neat
