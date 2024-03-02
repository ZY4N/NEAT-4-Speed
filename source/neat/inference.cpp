#include "neat/inference.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace neat::inference {

types::value_t activation_function(const types::value_t& signal) {
	return types::value_t{ 1.0 } / (types::value_t{ 2.0 } + std::exp(types::value_t{ -4.9 } * signal));
}

void evaluate_network_range(
	const types::network_group_t& network_group,
	debug_span<const types::value_t> network_inputs,
	debug_span<types::value_t> network_outputs,
	const neat::types::network_range_t& network_range
) {
	if (network_range.empty())
		return;

	assert(network_outputs.size() % network_group.networks.size() == 0);

	const auto num_inputs = network_inputs.size() / network_group.networks.size();
	const auto num_outputs = network_outputs.size() / network_group.networks.size();

	debug_vector<types::value_t> node_values;

	for (const auto& network_index : network_range.indices()) {
		const auto& network = network_group.networks[network_index];

		node_values.resize(num_inputs + network.incoming_connection_count_range.size());
		std::copy_n(&network_inputs[network_index * num_inputs], num_inputs, node_values.begin());
		// Only unconnected output nodes need to be set to zero but this is easier/hopefully faster.
		std::fill(node_values.begin() + num_inputs, node_values.end(), types::value_t{});

		auto network_conn_it = network_group.connections.begin() + network.incoming_connections_begin;
		const auto network_incoming_conn_counts = network.incoming_connection_count_range
													  .cspan(
														  network_group.incoming_connection_counts_and_node_lookups
													  );

		std::transform(
			network_incoming_conn_counts.begin(),
			network_incoming_conn_counts.end(),
			node_values.begin(),
			[&](const auto conn_count) {
				const auto activation = std::accumulate(
					network_conn_it,
					network_conn_it + conn_count,
					types::value_t{},
					[&node_values](const auto& sum, const auto& conn) {
						return sum + conn.weight * node_values[conn.source_node_index];
					}
				);
				network_conn_it += conn_count;
				return activation_function(activation);
			}
		);

		const auto output_node_lookup = std::span(network_group.incoming_connection_counts_and_node_lookups)
										 .subspan(network.incoming_connection_count_range.end(), num_outputs);

		std::transform(
			output_node_lookup.begin(),
			output_node_lookup.end(),
			&network_outputs[network_index * num_outputs],
			[&node_values](const auto& output_node_index) { return node_values[output_node_index]; }
		);
	}
}

} // namespace neat::inference
