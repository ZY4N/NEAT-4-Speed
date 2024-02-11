#include "neat/inference.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

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
	assert(network_outputs.size() % network_group.networks.size() == 0);

	const auto num_inputs = network_inputs.size() / network_group.networks.size();
	const auto num_outputs = network_outputs.size() / network_group.networks.size();

	debug_vector<types::value_t> node_values;
	node_values.resize(num_inputs + num_outputs + network_group.networks.front().nodes.size());

	for (const auto& network_index : network_range.indices()) {
		auto& network = network_group.networks[network_index];

		node_values.resize(num_inputs + num_outputs + network.nodes.size());
		std::copy_n(&network_inputs[num_inputs * network_index], num_inputs, node_values.begin());

		auto conn_it = network_group.connections.begin() + network.connections_begin;
		for (const auto& node : network.nodes.span<const types::node_t>(network_group.nodes)) {

			auto input_sum = 0.0f;
			const auto end_conn_it = conn_it + node.incoming_connection_count;
			while (conn_it != end_conn_it) {
				input_sum += conn_it->weight * node_values[conn_it->source_node_index];
				++conn_it;
			}

			node_values[node.node_index] = activation_function(input_sum);
		}

		std::copy_n(&node_values[num_inputs], num_outputs, &network_outputs[network_index * num_outputs]);
	}

}

} // namespace neat::inference
