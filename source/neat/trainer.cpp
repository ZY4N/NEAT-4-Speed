#include "neat/trainer.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream> // TODO remove
#include <numeric>
#include <thread>

namespace neat {

trainer::trainer(
	const evolution_config_t& evolution_config,
	const network_interface_config_t& network_interface_config,
	const std::size_t population_size,
	const std::uint32_t thread_count
) :
	m_evolution_config{ evolution_config },
	m_network_interface_config{ network_interface_config },
	m_population_size{ population_size },
	m_thread_count{ thread_count } {
	create_initial_population();
}

void trainer::create_initial_population() {

	auto& initial_population = m_populations[m_current_generation_index];

	initial_population.connections.clear();
	initial_population.connection_weights.clear();
	initial_population.connection_infos.clear();

	initial_population.networks.resize(m_population_size);
	for (auto& network : initial_population.networks) {
		network.hidden_node_count = 0;
		network.connections = types::conn_range_t::from_index_count(0, 0);
	}

	initial_population.species.resize(1);
	initial_population.species.front() = { .networks = types::network_range_t::from_index_count(0, m_population_size) };
}

void trainer::evolve(
	debug_span<const types::fitness_t> ancestor_fitness, inference::types::network_group_t& network_group
) {
	auto& ancestors = m_populations[m_current_generation_index];
	swap_population();
	auto& offspring = m_populations[m_current_generation_index];
	evolve_into(ancestors, ancestor_fitness, offspring);
	update_inference_network_group(network_group);
}

void trainer::swap_population() {
	m_current_generation_index = (m_current_generation_index + 1) % m_populations.size();
}

void trainer::mutate_all_connections(
	debug_span<const types::network_t> offspring_networks,
	debug_span<types::connection_weight_t> offspring_connection_weights,
	const types::network_range_t& conn_mutation_range
) {

	auto chance_distrib = std::uniform_real_distribution<float>(0.0f, 1.0f);
	auto weight_distrib = std::uniform_real_distribution<float>(
		m_evolution_config.weight_distribution_config.conn_weight_min,
		m_evolution_config.weight_distribution_config.conn_weight_max
	);
	auto weight_offset_distrib = std::uniform_real_distribution<float>(
		m_evolution_config.weight_distribution_config.conn_weight_offset_min,
		m_evolution_config.weight_distribution_config.conn_weight_offset_max
	);

	for (const auto& offspring_network : conn_mutation_range.span(offspring_networks)) {
		for (auto& weight : offspring_network.connections.span(offspring_connection_weights)) {
			if (chance_distrib(m_rng) < m_evolution_config.mutation_rate_config.uniform_mutation_rate) {
				weight += weight_offset_distrib(m_rng);
			} else {
				weight = weight_distrib(m_rng);
			}
		}
	}
}

void trainer::mutate_some_connections(
	debug_span<const types::network_t> offspring_networks,
	debug_span<types::connection_weight_t> offspring_connection_weights,
	const types::network_range_t& conn_mutation_range
) {

	auto chance_distrib = std::uniform_real_distribution<float>(0.0f, 1.0f);
	auto weight_distrib = std::uniform_real_distribution<float>(
		m_evolution_config.weight_distribution_config.conn_weight_min,
		m_evolution_config.weight_distribution_config.conn_weight_max
	);

	auto weight_offset_distrib = std::uniform_real_distribution<float>(
		m_evolution_config.weight_distribution_config.conn_weight_offset_min,
		m_evolution_config.weight_distribution_config.conn_weight_offset_max
	);

	for (const auto& offspring_network : conn_mutation_range.span(offspring_networks)) {
		if (chance_distrib(m_rng) < m_evolution_config.mutation_rate_config.network_mutation_rate) {
			for (auto& weight : offspring_network.connections.span(offspring_connection_weights)) {
				if (chance_distrib(m_rng) < m_evolution_config.mutation_rate_config.uniform_mutation_rate) {
					weight += weight_offset_distrib(m_rng);
				} else {
					weight = weight_distrib(m_rng);
				}
			}
		}
	}
}

void trainer::apply_add_conn_mutations(
	debug_span<const types::network_t> ancestor_networks,
	debug_span<const types::connection_t> ancestor_connections,
	debug_span<const types::network_index_t> ancestor_lookup,
	debug_span<types::network_t> offspring_networks,
	debug_span<types::connection_t> offspring_connections,
	debug_span<types::connection_weight_t> offspring_connection_weights,
	debug_span<types::connection_info_t> offspring_connection_infos,
	const types::network_range_t& add_conn_mutation_range
) {
	debug_vector<bool> dst_nodes_taken;
	debug_vector<types::node_index_t> node_stack;

	auto weight_distrib = std::uniform_real_distribution<types::connection_weight_t>(
		m_evolution_config.weight_distribution_config.conn_weight_min,
		m_evolution_config.weight_distribution_config.conn_weight_max
	);

	for (const auto& network_index : add_conn_mutation_range.indices()) {

		const auto& ancestor_network = ancestor_networks[ancestor_lookup[network_index]];
		const auto& ancestor_network_connections = ancestor_network.connections.span(ancestor_connections);

		auto& offspring_network = offspring_networks[network_index];
		const auto& offspring_network_connections = offspring_network.connections.span(offspring_connections);
		const auto& offspring_network_connection_weights = offspring_network.connections.span(
			offspring_connection_weights
		);

		const auto num_src_nodes = m_network_interface_config.input_count + ancestor_network.hidden_node_count;
		const auto num_dst_nodes = m_network_interface_config.output_count + ancestor_network.hidden_node_count;
		dst_nodes_taken.resize(num_dst_nodes);

		using node_dist_t = std::uniform_int_distribution<types::node_index_t>;

		auto from_distrib = node_dist_t(0, num_src_nodes - 1);
		auto to_distrib = node_dist_t(
			m_network_interface_config.input_count,
			m_network_interface_config.input_count + num_dst_nodes - 1
		);

		auto from_index{ invalid_node_index }, to_index{ invalid_node_index };

		// TODO This number doesn't have a deeper meaning. Fix it.
		auto tries_left = num_src_nodes * num_dst_nodes + 1;
		auto connection_valid = false;
		while (tries_left--) {
			from_index = from_distrib(m_rng);
			// input | output | hidden
			// -> output indices need to be skipped.
			if (from_index >= m_network_interface_config.input_count) {
				from_index += m_network_interface_config.output_count;
			}

			// TODO remove all num's

			std::fill(dst_nodes_taken.begin(), dst_nodes_taken.end(), false);
			for (const auto& conn : ancestor_network_connections) {
				if (conn.from == from_index) {
					const auto lookup_index = conn.to - m_network_interface_config.input_count;
					assert(lookup_index < dst_nodes_taken.size());
					dst_nodes_taken[lookup_index] = true;
				}
			}

			const auto num_dst_nodes_taken = std::accumulate(
				dst_nodes_taken.begin(),
				dst_nodes_taken.end(),
				types::node_index_t{}
			);

			if (num_dst_nodes_taken == num_dst_nodes) {
				continue;
			}

			// If the src node is hidden and fully connected, it is obviously not connected to itself, which is
			// addressed here.
			if (((num_dst_nodes_taken + 1) == num_dst_nodes and from_index >= m_network_interface_config.input_count and
			     not dst_nodes_taken[from_index - m_network_interface_config.input_count])) {
				continue;
			}

			do {
				to_index = to_distrib(m_rng);
			} while (from_index == to_index or dst_nodes_taken[to_index - m_network_interface_config.input_count]);

			if (would_create_loop(ancestor_connections, ancestor_network, from_index, to_index, node_stack)) {
				continue;
			}

			connection_valid = true;
			break;
		}

		if (connection_valid) {
			// The connections span was already extended to include one uninitialized connection.
			// This connection can now be set.
			assert(from_index < 140'736);
			assert(to_index < 140'736);
			assert(
				from_index < m_network_interface_config.input_count or
				(from_index >= (m_network_interface_config.input_count + m_network_interface_config.output_count) and
			     from_index < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			                   offspring_network.hidden_node_count))
			);
			assert(to_index >= m_network_interface_config.input_count);
			assert(
				to_index < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			                offspring_network.hidden_node_count)
			);

			offspring_network_connections.back() = { .from = from_index, .to = to_index };
			offspring_network_connection_weights.back() = weight_distrib(m_rng);
			m_conn_lookup.update_connection_info(
				offspring_connection_infos,
				offspring_network.connections.end() - 1,
				from_index,
				to_index
			);
		} else {
			// If no connections can be added give up, and leave the network unmodified.
			--offspring_network.connections.end();
		}
	}
}

void trainer::apply_add_node_mutations(
	debug_span<const types::network_t> ancestor_networks,
	debug_span<const types::connection_t> ancestor_connections,
	debug_span<const types::connection_weight_t> ancestor_connection_weights,
	debug_span<const types::network_index_t> ancestor_lookup,
	debug_span<types::network_t> offspring_networks,
	debug_span<types::connection_t> offspring_connections,
	debug_span<types::connection_weight_t> offspring_connection_weights,
	debug_span<types::connection_info_t> offspring_connection_infos,
	const types::network_range_t& add_node_mutation_range
) {
	for (const auto& network_index : add_node_mutation_range.indices()) {

		const auto& ancestor_network = ancestor_networks[ancestor_lookup[network_index]];
		const auto& ancestor_network_connections = ancestor_network.connections.span(ancestor_connections);

		auto& offspring_network = offspring_networks[network_index];
		const auto& offspring_network_connections = offspring_network.connections.span(offspring_connections);
		const auto& offspring_network_connection_weights = offspring_network.connections.span(
			offspring_connection_weights
		);
		const auto& offspring_network_connection_infos = offspring_network.connections.span(offspring_connection_infos);

		if (ancestor_network.connections.empty()) {
			// TODO Well that's a bit of a problem isn't it!!
			offspring_network.connections.end() -= 2;
			--offspring_network.hidden_node_count;
			continue;
		}

		// Select random connection to be split
		auto conn_distrib = std::uniform_int_distribution<types::conn_index_t>(
			0,
			ancestor_network_connections.size() - 1
		);
		const auto split_connection_index = conn_distrib(m_rng);
		const auto& split_connection = ancestor_network_connections[split_connection_index];
		const auto& split_connection_weight = ancestor_connection_weights[split_connection_index];

		// Deactivate old connection.
		offspring_network_connection_infos[split_connection_index].enabled = false;

		// Node has already been "added" to the end while counting connections
		const auto added_node =
			(m_network_interface_config.input_count + m_network_interface_config.output_count +
		     offspring_network.hidden_node_count - 1);

		const auto incoming_conn_index = offspring_network.connections.size() - 2;
		const auto outgoing_conn_index = offspring_network.connections.size() - 1;

		// Create new connections:

		assert(added_node != 0);
		assert(added_node < 140'736);
		assert(
			split_connection.to >= m_network_interface_config.input_count and
			split_connection.to < (m_network_interface_config.input_count + m_network_interface_config.output_count +
		                           ancestor_network.hidden_node_count)
		);

		assert(split_connection.from < 140'736);
		assert(split_connection.to < 140'736);

		offspring_network_connections[incoming_conn_index] = { .from = split_connection.from, .to = added_node };
		offspring_network_connections[outgoing_conn_index] = { .from = added_node, .to = split_connection.to };

		offspring_network_connection_weights[incoming_conn_index] = split_connection_weight;
		offspring_network_connection_weights[outgoing_conn_index] = 1.0f;

		m_conn_lookup.update_connection_info(
			offspring_connection_infos,
			offspring_network.connections.begin() + incoming_conn_index,
			offspring_network_connections[incoming_conn_index].from,
			offspring_network_connections[incoming_conn_index].to
		);

		m_conn_lookup.update_connection_info(
			offspring_connection_infos,
			offspring_network.connections.begin() + outgoing_conn_index,
			offspring_network_connections[outgoing_conn_index].from,
			offspring_network_connections[outgoing_conn_index].to
		);
	}
}

void trainer::create_crossovers(
	const debug_span<const types::network_t> ancestor_networks,
	const debug_span<const types::connection_t> ancestor_connections,
	const debug_span<const types::connection_weight_t> ancestor_connection_weights,
	const debug_span<const types::connection_info_t> ancestor_connection_infos,
	const debug_span<const types::fitness_t> ancestor_fitness,
	const debug_span<const types::parents_t> parents_lookup,
	const debug_span<const seed_t> disjoint_excess_conn_selection_seeds,
	const debug_span<types::network_t> offspring_networks,
	const debug_span<types::connection_t> offspring_connections,
	const debug_span<types::connection_weight_t> offspring_connection_weights,
	const debug_span<types::connection_info_t> offspring_connection_infos,
	const std::size_t seed_offset,
	const types::network_range_t& crossover_range
) {
	auto parents_connection_selection_distrib = std::uniform_int_distribution<std::uint8_t>(false, true);
	auto parents_connection_survival_distrib = std::uniform_int_distribution<std::uint8_t>(false, true);
	auto chance_distrib = std::uniform_real_distribution<float>(0.0f, 1.0f);

	std::default_random_engine disjoint_and_excess_conn_selection_engine{ 0 };

	// The crossover_range needs to be indexed relative because the parents_lookup is already offset.
	for (types::network_index_t i{}; i != crossover_range.size(); ++i) {

		static constexpr auto num_parents = std::size_t{ 2 };

		// Reset m_rng to replicate the decision done while counting the crossover connections.
		disjoint_and_excess_conn_selection_engine.seed(disjoint_excess_conn_selection_seeds[seed_offset + i]);
		parents_connection_survival_distrib.reset();

		const auto& parent_indices = parents_lookup[i];

		std::array<types::conn_range_t, num_parents> parent_connection_its;
		std::transform(
			parent_indices.begin(),
			parent_indices.end(),
			parent_connection_its.begin(),
			[&](const auto& parent_index) { return ancestor_networks[parent_index].connections; }
		);

		auto& offspring_network = offspring_networks[crossover_range.begin() + i];

		auto offspring_network_connections = offspring_network.connections.span(offspring_connections);
		auto offspring_network_connection_weights = offspring_network.connections.span(offspring_connection_weights);
		auto offspring_network_connection_infos = offspring_network.connections.span(offspring_connection_infos);

		auto fit_index = types::network_index_t{ 0 }, unfit_index = types::network_index_t{ 1 };

		std::array<types::fitness_t, num_parents> parent_fitness;
		std::transform(
			parent_indices.begin(),
			parent_indices.end(),
			parent_fitness.begin(),
			[&ancestor_fitness](const auto& parent_index) { return ancestor_fitness[parent_index]; }
		);

		const auto matching_fitness = does_fitness_match(parent_fitness[fit_index], parent_fitness[unfit_index]);
		if (not matching_fitness and parent_fitness[fit_index] < parent_fitness[unfit_index]) {
			std::swap(fit_index, unfit_index);
		}

		types::conn_index_t offspring_network_connection_index{};
		types::node_index_t offspring_max_node_index{ m_network_interface_config.input_count +
			                                          m_network_interface_config.output_count };

		const auto add_connection = [&](const auto& inherit_connection_index, const bool either_parent_deactivated) {
			const auto inherited_conn_index = parent_connection_its[inherit_connection_index].begin();

			assert(ancestor_connections[inherited_conn_index].from != ancestor_connections[inherited_conn_index].to);

			auto& offspring_connection = offspring_network_connections
				[offspring_network_connection_index] = ancestor_connections[inherited_conn_index];

			offspring_network_connection_weights[offspring_network_connection_index] = ancestor_connection_weights
				[inherited_conn_index];

			auto& offspring_connection_info = offspring_network_connection_infos
				[offspring_network_connection_index] = ancestor_connection_infos[inherited_conn_index];

			offspring_connection_info.enabled = not(
				either_parent_deactivated and
				chance_distrib(m_rng) < m_evolution_config.mutation_rate_config.keep_disabled_rate
			);

			offspring_max_node_index = std::max(
				{ offspring_max_node_index, offspring_connection.from, offspring_connection.to }
			);

			assert(offspring_max_node_index < 140'736);
			assert(offspring_connection.from != offspring_connection.to);

			++offspring_network_connection_index;
		};

		while (std::none_of(
			parent_connection_its.begin(),
			parent_connection_its.end(),
			[](const auto& parent_connection) { return parent_connection.empty(); }
		)) {

			auto fit_conn_index = parent_connection_its[fit_index].begin();
			auto unfit_conn_index = parent_connection_its[unfit_index].begin();

			const auto& fit_inno_num = ancestor_connection_infos[fit_conn_index].innovation_number;
			const auto& unfit_inno_num = ancestor_connection_infos[unfit_conn_index].innovation_number;

			auto inherit_connection_index = invalid_conn_index;
			bool either_parent_conn_deactivated{};

			if (fit_inno_num == unfit_inno_num) { // matching gene
				inherit_connection_index = parents_connection_selection_distrib(m_rng);
				++fit_conn_index;
				++unfit_conn_index;
			} else {
				const auto fit_connection_next = fit_inno_num < unfit_inno_num;
				if (matching_fitness) { // matching fitness and matching gene
					types::conn_index_t next_connection;
					if (fit_connection_next) {
						next_connection = fit_index;
					} else {
						next_connection = unfit_index;
					}
					// A described before, if this was not done in a deterministic way,
					// the number of offspring connections would not be predictable.

					if (parents_connection_survival_distrib(disjoint_and_excess_conn_selection_engine)) {
						inherit_connection_index = next_connection;
						either_parent_conn_deactivated = not(
							ancestor_connection_infos[parent_connection_its[fit_index].begin()].enabled and
							ancestor_connection_infos[parent_connection_its[unfit_index].begin()].enabled
						);
					}
				} else if (fit_connection_next) { // fit matching gene
					inherit_connection_index = fit_index;
					either_parent_conn_deactivated = ancestor_connection_infos[parent_connection_its[fit_index].begin()]
														 .enabled;
				}
				if (fit_connection_next) { // fit disjoint gene
					++fit_conn_index;
				} else { // unfit disjoint gene
					++unfit_conn_index;
				}
			}
			if (inherit_connection_index != invalid_conn_index) {
				add_connection(inherit_connection_index, either_parent_conn_deactivated);
			}
			parent_connection_its[fit_index].begin() = fit_conn_index;
			parent_connection_its[unfit_index].begin() = unfit_conn_index;
		}

		// Handler excess genes

		const auto excess_range_it = std::find_if(
			parent_connection_its.begin(),
			parent_connection_its.end(),
			[](const auto& parent_connection) { return not parent_connection.empty(); }
		);

		const auto excess_parent_index = static_cast<types::network_index_t>(
			excess_range_it - parent_connection_its.begin()
		);

		if (excess_parent_index == fit_index or (excess_parent_index == unfit_index and matching_fitness)) {
			while (not excess_range_it->empty()) {
				add_connection(excess_parent_index, ancestor_connection_infos[excess_range_it->begin()].enabled);
				++excess_range_it->begin();
			}
		}

		// Set the node count.
		offspring_network.hidden_node_count = offspring_max_node_index + 1 -
			(m_network_interface_config.input_count + m_network_interface_config.output_count);
		assert(offspring_network.hidden_node_count < 140'736);

		assert(offspring_network_connection_index == offspring_network.connections.size());
	}
}

std::size_t trainer::crossover_offspring_conn_count(
	std::default_random_engine& disjoint_and_excess_connection_selection_engine,
	std::uniform_int_distribution<std::uint8_t>& parents_connection_survival_distrib,
	debug_span<const types::network_t> ancestor_networks,
	debug_span<const types::connection_info_t> ancestor_connection_infos,
	debug_span<const types::fitness_t> ancestor_fitness,
	const types::parents_t& parent_indices
) {

	static constexpr auto num_parents = std::size_t{ 2 };

	std::array<types::conn_range_t, num_parents> parent_connection_its;
	std::transform(
		parent_indices.begin(),
		parent_indices.end(),
		parent_connection_its.begin(),
		[&](const auto& parent_index) {
			assert(parent_index < ancestor_networks.size());
			return ancestor_networks[parent_index].connections;
		}
	);

	auto fit_index = types::network_index_t{ 0 }, unfit_index = types::network_index_t{ 1 };

	std::array<types::fitness_t, num_parents> parent_fitness;
	std::transform(
		parent_indices.begin(),
		parent_indices.end(),
		parent_fitness.begin(),
		[&ancestor_fitness](const auto& parent_index) { return ancestor_fitness[parent_index]; }
	);

	const auto matching_fitness = does_fitness_match(parent_fitness[fit_index], parent_fitness[unfit_index]);
	if (not matching_fitness and parent_fitness[fit_index] < parent_fitness[unfit_index]) {
		std::swap(fit_index, unfit_index);
	}

	types::conn_index_t offspring_network_connection_count{};

	while (std::none_of(parent_connection_its.begin(), parent_connection_its.end(), [](const auto& parent_connection) {
		return parent_connection.empty();
	})) {
		auto& fit_conn_index = parent_connection_its[fit_index].begin();
		auto& unfit_conn_index = parent_connection_its[unfit_index].begin();

		const auto& fit_inno_num = ancestor_connection_infos[fit_conn_index].innovation_number;
		const auto& unfit_inno_num = ancestor_connection_infos[unfit_conn_index].innovation_number;

		if (fit_inno_num == unfit_inno_num) { // matching gene
			++offspring_network_connection_count;
			++fit_conn_index;
			++unfit_conn_index;
		} else {
			const auto fit_connection_next = fit_inno_num < unfit_inno_num;
			if (matching_fitness) { // matching fitness and matching gene
				// A described before, if this was not done in a deterministic way,
				// the number of offspring connections would not be predictable.
				if (parents_connection_survival_distrib(disjoint_and_excess_connection_selection_engine)) {
					++offspring_network_connection_count;
				}
			} else if (fit_connection_next) { // fit matching gene
				++offspring_network_connection_count;
			}
			if (fit_connection_next) { // fit disjoint gene
				++fit_conn_index;
			} else { // unfit disjoint gene
				++unfit_conn_index;
			}
		}
	}

	// Handler excess genes

	const auto excess_range_it = std::find_if(
		parent_connection_its.begin(),
		parent_connection_its.end(),
		[](const auto& parent_connection) { return not parent_connection.empty(); }
	);

	const auto excess_parent_index = static_cast<types::network_index_t>(
		excess_range_it - parent_connection_its.begin()
	);
	if (excess_parent_index == fit_index or (excess_parent_index == unfit_index and matching_fitness)) {
		offspring_network_connection_count += excess_range_it->size();
	}

	return offspring_network_connection_count;
}

void trainer::calc_species_fitness(
	debug_span<const types::species_t> all_species,
	debug_span<const types::fitness_t> network_fitness,
	debug_span<float> species_fitness,
	const types::species_range_t& species_range
) {
	const auto in_range_species = species_range.span(all_species);
	const auto in_range_fitness = species_range.span(species_fitness);
	std::transform(
		in_range_species.begin(),
		in_range_species.end(),
		in_range_fitness.begin(),
		[&network_fitness](const auto& species) {
			const auto species_scores = species.networks.span(network_fitness);
			const auto total_species_score = std::accumulate(species_scores.begin(), species_scores.end(), 0.0f);
			return total_species_score / static_cast<float>(species_scores.size());
		}
	);
}

void trainer::divide_offspring_between_species(
	debug_span<const float> ancestor_species_fitness, debug_vector<types::network_index_t>& species_offspring_counts
) {
	// TODO put into class.
	static debug_vector<std::pair<types::species_index_t, float>> flooring_errors;
	flooring_errors.resize(ancestor_species_fitness.size());

	const auto [fitness_min_it, fitness_max_it] = std::minmax_element(
		ancestor_species_fitness.begin(),
		ancestor_species_fitness.end()
	);

	const auto fitness_offset = -*fitness_min_it;
	const auto fitness_range = *fitness_max_it - *fitness_min_it;
	const auto fitness_scale = fitness_range == 0.0f ? 0.0f : (1.0f / fitness_range);
	const auto adjust_fitness = [fitness_offset, fitness_scale](const float fitness) {
		return fitness_scale * (fitness + fitness_offset);
	};

	auto fitness_sum = types::fitness_t{};
	for (const auto& species_fitness : ancestor_species_fitness) {
		fitness_sum += adjust_fitness(species_fitness);
	}

	auto num_offspring = types::network_index_t{};
	const auto population_scale = static_cast<float>(m_population_size);

	std::cout << "Fitness sum: " << fitness_sum << std::endl;

	if (fitness_sum == 0.0f) {
		const auto species_range = types::network_range_t::from_index_count(0, m_population_size);
		auto species_index = std::size_t{};
		for (const auto& segment : species_range.balanced_segments(species_offspring_counts.size())) {
			species_offspring_counts[species_index++] = segment.size();
		}

	} else {
		for (std::size_t i{}; i != ancestor_species_fitness.size(); ++i) {
			std::cout << "species_fitness[" << i << "]: " << adjust_fitness(ancestor_species_fitness[i]) << std::endl;

			const auto none_negative_fitness = adjust_fitness(ancestor_species_fitness[i]);
			const auto portion = none_negative_fitness / fitness_sum;
			std::cout << "portion: " << portion << std::endl;

			const auto population_portion = portion * population_scale;
			const auto min_portion = std::floor(population_portion);

			flooring_errors[i] = { i, population_portion - min_portion };

			const auto int_min_portion = static_cast<types::network_index_t>(min_portion);
			num_offspring += int_min_portion;

			species_offspring_counts[i] = int_min_portion;
		}

		// Identify largest flooring errors.
		std::sort(
			flooring_errors.begin(),
			flooring_errors.end(),
			[](const auto& index_error_a, const auto& index_error_b) {
				return index_error_a.second > index_error_b.second;
			}
		);

		// Give extra networks to species with big flooring error.
		// TODO the min only fixes the symptom, not the actual issue.
		const auto num_offspring_missing = std::min(m_population_size - num_offspring, flooring_errors.size());
		for (std::size_t i{}; i != num_offspring_missing; ++i) {
			++species_offspring_counts[flooring_errors[i].first];
		}
	}
}

void trainer::calculate_species_offspring_composition_and_sample_ancestors(
	const debug_span<const types::species_t> all_ancestor_species,
	const debug_span<const types::network_t> all_ancestor_networks,
	const debug_span<const float> ancestor_species_fitness,
	const debug_span<const types::network_index_t> species_offspring_counts,
	const debug_span<types::population_composition_t> offspring_species_composition,
	const debug_span<debug_vector<types::network_index_t>> species_ancestor_lookups,
	const types::species_range_t& species_range
) {
	for (const auto& species_index : species_range.indices()) {

		const auto& ancestor_species = all_ancestor_species[species_index];

		const auto species_fitness = ancestor_species.networks.span(ancestor_species_fitness);

		//-----------------------[ ancestor extinction ]-----------------------//

		// The number of extinct ancestors is limited by two metrics.
		// One limits by ancestor population portion (So only a small portion goes extinct)
		// The limits by ancestor fitness (So only ancestor with low fitness go extinct)

		debug_vector<std::pair<types::network_index_t, float>> sorted_network_scores(species_fitness.size());
		for (std::size_t i{}; i != species_fitness.size(); ++i) {
			sorted_network_scores[i] = { i, species_fitness[i] };
		}

		std::sort(
			sorted_network_scores.begin(),
			sorted_network_scores.end(),
			[](const auto& index_score_a, const auto& index_score_b) {
				return index_score_a.second < index_score_b.second;
			}
		);

		const auto min_score = sorted_network_scores.front().second;
		const auto max_score = sorted_network_scores.back().second;
		const auto score_threshold = std::lerp(
			min_score,
			max_score,
			m_evolution_config.extinction_config.max_remove_score_portion
		);

		const auto max_removed_score_portion = static_cast<std::uint32_t>(
			std::lower_bound(
				sorted_network_scores.begin(),
				sorted_network_scores.end(),
				score_threshold,
				[](const auto& index_score, const auto& threshold) { return index_score.second < threshold; }
			) -
			sorted_network_scores.end()
		);

		const auto max_removed_population_portion = static_cast<types::network_index_t>(std::round(
			m_evolution_config.extinction_config.max_remove_population_portion *
			static_cast<float>(ancestor_species.networks.size())
		));

		const auto num_extinct = std::min(max_removed_population_portion, max_removed_score_portion);

		const auto ancestor_indices_scores = debug_span{ sorted_network_scores.begin() + num_extinct,
			                                             sorted_network_scores.end() };

		auto species_offspring_count = species_offspring_counts[species_index];
		auto& [add_conn_mutation_count, add_node_mutation_count, conn_mutation_count, champion_count, crossover_count] =
			offspring_species_composition[species_index];

		//-----------------------[ champion count ]-----------------------//

		// If species is big enough, keep best performing network unchanged
		auto champion = std::optional<types::network_index_t>{ std::nullopt };
		champion_count = 0;
		if (species_offspring_count > 0 and
		    ancestor_indices_scores.size() >= m_evolution_config.mutation_rate_config.min_network_champion_size) {
			const auto champion_index = std::max_element(species_fitness.begin(), species_fitness.end()) -
				species_fitness.begin();
			std::cout << "champion_index: " << champion_index << std::endl;
			champion = champion_index;
			++champion_count;
			species_offspring_count--;
		}

		const auto calc_population_portion_count = [&](const double rate) {
			std::binomial_distribution<std::size_t> distrib(species_offspring_count, rate);
			return distrib(m_rng);
		};

		//-----------------------[ mutation count ]-----------------------//

		// Calculate offspring portion that gets mutated.
		auto mutation_count = static_cast<types::network_index_t>(std::round(
			m_evolution_config.mutation_rate_config.offspring_mutation_rate *
			static_cast<float>(species_offspring_count)
		));

		std::cout << "mutation count: " << mutation_count << std::endl;

		add_conn_mutation_count = calc_population_portion_count(
			m_evolution_config.mutation_rate_config.new_connection_rate
		);
		add_node_mutation_count = calc_population_portion_count(m_evolution_config.mutation_rate_config.new_node_rate);

		auto topological_mutation_count = add_conn_mutation_count + add_node_mutation_count;
		// This addresses the case that statistical outliers could lead to exceeding count.
		{
			auto remove_conn_mutation_next = m_evolution_config.mutation_rate_config.new_connection_rate >
				m_evolution_config.mutation_rate_config.new_node_rate;
			while (topological_mutation_count > mutation_count) {
				// This far from perfect, but it might be good enough...
				if (remove_conn_mutation_next) {
					if (add_conn_mutation_count) {
						--add_conn_mutation_count;
						--topological_mutation_count;
					}
				} else {
					if (add_node_mutation_count) {
						--add_node_mutation_count;
						--topological_mutation_count;
					}
				}
				remove_conn_mutation_next = !remove_conn_mutation_next;
			}
		}

		// All other mutations are simple weight mutations.
		conn_mutation_count = mutation_count - topological_mutation_count;

		//-----------------------[ crossover count ]-----------------------//

		// A small portion is created by cross species offspring.
		auto inter_species_crossover_count = all_ancestor_species.size() == 1
			? 0
			: calc_population_portion_count(m_evolution_config.mutation_rate_config.inter_species_mating_rate);

		auto everything_but_in_species_crossovers_count = (mutation_count + inter_species_crossover_count);

		while (everything_but_in_species_crossovers_count > species_offspring_count) {
			// This will fail if the mutation are already too many without the inter-species-crossovers.
			assert(inter_species_crossover_count);
			--inter_species_crossover_count;
			--everything_but_in_species_crossovers_count;
		}

		auto in_species_crossovers_count = species_offspring_count - everything_but_in_species_crossovers_count;
		if (ancestor_indices_scores.size() == 1) {
			if (all_ancestor_species.size() > 1) {
				inter_species_crossover_count += in_species_crossovers_count;
			} else {
				add_conn_mutation_count += in_species_crossovers_count;
			}
			in_species_crossovers_count = 0;
		}

		crossover_count = inter_species_crossover_count + in_species_crossovers_count;

		//-----------------------[ ancestor selection ]-----------------------//

		// The crossovers have two parent networks, so add that to the total count.
		const auto ancestor_count = species_offspring_counts[species_index] + crossover_count;

		// Now sample the ancestor indices
		auto& species_ancestor_lookup = species_ancestor_lookups[species_index];
		species_ancestor_lookup.resize(ancestor_count);

		auto ancestor_lookup_it = species_ancestor_lookup.begin();

		std::uniform_int_distribution<types::network_index_t> in_species_ancestor_index_distrib(
			0,
			ancestor_indices_scores.size() - 1
		);
		std::uniform_int_distribution<types::network_index_t> all_ancestor_index_distrib(
			0,
			all_ancestor_networks.size() - 1
		);

		const auto mutated_ancestor_count = (add_conn_mutation_count + add_node_mutation_count + conn_mutation_count);

		//-----------------------[ mutation ancestors ]-----------------------//

		std::cout << "mutated_ancestor_count: " << mutated_ancestor_count << std::endl;
		std::generate(ancestor_lookup_it, ancestor_lookup_it + mutated_ancestor_count, [&]() {
			const auto ancestor = in_species_ancestor_index_distrib(m_rng);
			return ancestor;
		});
		ancestor_lookup_it += mutated_ancestor_count;

		// Insert champion index.
		if (champion) {
			*ancestor_lookup_it++ = champion.value();
		}

		//-----------------------[ in-species-crossover parents ]-----------------------//

		auto parent_it = reinterpret_cast<types::parents_t*>(ancestor_lookup_it.base());

		std::cout << "in_species_crossovers_count: " << in_species_crossovers_count << std::endl;
		std::generate(parent_it, parent_it + in_species_crossovers_count, [&]() -> types::parents_t {
			const auto parent_a_rng = in_species_ancestor_index_distrib(m_rng);
			assert(parent_a_rng < ancestor_indices_scores.size());
			auto parent_index_a = ancestor_indices_scores[parent_a_rng].first;

			types::network_index_t parent_index_b;
			do {
				parent_index_b = ancestor_indices_scores[in_species_ancestor_index_distrib(m_rng)].first;
			} while (parent_index_a == parent_index_b);

			return { parent_index_a, parent_index_b };
		});
		parent_it += in_species_crossovers_count;

		//-----------------------[ inter-species-crossover parents ]-----------------------//

		std::cout << "inter_species_crossover_count: " << inter_species_crossover_count << std::endl;

		std::generate(parent_it, parent_it + inter_species_crossover_count, [&]() -> types::parents_t {
			auto in_species_parent = in_species_ancestor_index_distrib(m_rng);

			types::network_index_t external_species_parent;
			do {
				external_species_parent = all_ancestor_index_distrib(m_rng);
			} while (ancestor_species.networks.contains(external_species_parent));
			return { in_species_parent, external_species_parent };
		});
		parent_it += inter_species_crossover_count;
	}
}

template<typename T>
void copy_connection_data(
	debug_span<const types::network_index_t> ancestor_lookup,
	debug_span<const types::network_t> ancestor_networks,
	debug_span<const types::network_t> offspring_networks,
	debug_span<const T> ancestor_connection_data,
	debug_span<T> offspring_connection_data,
	const types::network_range_t& network_range
) {
	for (const auto& network_index : network_range.indices()) {
		const auto& ancestor_network = ancestor_networks[ancestor_lookup[network_index]];
		const auto& offspring_network = offspring_networks[network_index];

		const auto data_src = ancestor_network.connections.span<const T>(ancestor_connection_data);
		const auto data_dst = offspring_network.connections.span<T>(offspring_connection_data);

		std::copy(data_src.begin(), data_src.end(), data_dst.begin());
	}
}

void trainer::evolve_into(
	const types::population_t& ancestors,
	debug_span<const types::fitness_t> ancestor_fitness,
	types::population_t& offspring
) {
	debug_vector<std::thread> threads;
	threads.reserve(m_thread_count);

	const auto ancestor_species_range = types::species_range_t::from_range(ancestors.species);
	const auto ancestor_species_thread_segments = ancestor_species_range.balanced_segments(m_thread_count);

	std::cout << "|-------------[ calc_species_fitness ]-------------|" << std::endl;

	debug_vector<float> ancestor_species_fitness(ancestors.species.size());
	for (const auto& species_segment : ancestor_species_thread_segments) {
		threads.emplace_back([&, species_segment]() {
			calc_species_fitness(ancestors.species, ancestor_fitness, ancestor_species_fitness, species_segment);
		});
	}

	for (auto& thread : threads) {
		thread.join();
	}
	threads.clear();

	std::cout << "|-------------[ divide_offspring_between_species ]-------------|" << std::endl;

	// Calculate every species portion of next generation proportional to species ancestor_fitness.
	debug_vector<types::network_index_t> species_offspring_counts(ancestors.species.size());
	divide_offspring_between_species(ancestor_species_fitness, species_offspring_counts);

	debug_vector<types::population_composition_t> species_offspring_compositions(ancestors.species.size());
	debug_vector<debug_vector<types::network_index_t>> species_ancestor_lookups(ancestors.species.size());

	std::cout << "|-------------[ calculate_species_offspring_composition_and_sample_ancestors ]-------------|"
			  << std::endl;

	for (const auto& species_segment : ancestor_species_thread_segments) {
		threads.emplace_back([&, species_segment]() {
			calculate_species_offspring_composition_and_sample_ancestors(
				ancestors.species,
				ancestors.networks,
				ancestor_fitness,
				species_offspring_counts,
				species_offspring_compositions,
				species_ancestor_lookups,
				species_segment
			);
		});
	}

	for (auto& thread : threads) {
		thread.join();
	}
	threads.clear();

	offspring.networks.resize(ancestors.networks.size());

	std::cout << "|-------------[ accumulate species compositions ]-------------|" << std::endl;

	// Add all compositions together
	types::population_composition_t offspring_composition{};
	for (std::size_t i{}; i != species_offspring_compositions.size(); ++i) {
		const auto& species_offspring_composition = species_offspring_compositions[i];
		offspring_composition.add_conn_mutation_count += species_offspring_composition.add_conn_mutation_count;
		offspring_composition.add_node_mutation_count += species_offspring_composition.add_node_mutation_count;
		offspring_composition.champion_count += species_offspring_composition.champion_count;
		offspring_composition.conn_mutation_count += species_offspring_composition.conn_mutation_count;
		offspring_composition.crossover_count += species_offspring_composition.crossover_count;
	}

	std::cout << "add_conn_mutation_count: " << offspring_composition.add_conn_mutation_count << std::endl;
	std::cout << "add_node_mutation_count: " << offspring_composition.add_node_mutation_count << std::endl;
	std::cout << "conn_mutation_count: " << offspring_composition.conn_mutation_count << std::endl;
	std::cout << "champion_count: " << offspring_composition.champion_count << std::endl;
	std::cout << "crossover_count: " << offspring_composition.crossover_count << std::endl;

	// Combine ancestor lookup
	const auto directly_inherited_ancestor_count =
		(offspring_composition.add_conn_mutation_count + offspring_composition.add_node_mutation_count +
	     offspring_composition.conn_mutation_count + offspring_composition.champion_count);

	const auto ancestors_count = (directly_inherited_ancestor_count + offspring_composition.crossover_count * 2);

	std::cout << "|-------------[ Copy ancestors into one vector ]-------------|" << std::endl;

	debug_vector<types::network_index_t> ancestor_lookup(ancestors_count);

	// Copy all indices into this continuous vector.
	auto directly_inherited_ancestor_lookup_index = types::network_index_t{};
	auto crossover_ancestor_indices_offset = directly_inherited_ancestor_count;

	// Has to be single threaded, because offset is only calculated while copying
	for (std::size_t i{}; i != ancestors.species.size(); ++i) {

		const auto& species_ancestor_lookup = species_ancestor_lookups[i];
		const auto& species_offspring_composition = species_offspring_compositions[i];

		const auto add_ancestors = [&](types::network_index_t& dst_index,
		                               types::network_index_t& src_index,
		                               const types::network_index_t count) {
			const auto src_end = src_index + count;
			while (src_index != src_end) {
				ancestor_lookup[dst_index++] = species_ancestor_lookup[src_index++];
			}
		};

		auto species_indices_index = types::network_index_t{};

		add_ancestors(
			directly_inherited_ancestor_lookup_index,
			species_indices_index,
			species_offspring_composition.add_conn_mutation_count +
				species_offspring_composition.add_node_mutation_count +
				species_offspring_composition.conn_mutation_count + species_offspring_composition.champion_count
		);

		add_ancestors(
			crossover_ancestor_indices_offset,
			species_indices_index,
			species_offspring_composition.crossover_count * (sizeof(types::parents_t) / sizeof(types::network_index_t))
		);
	}

	// Create spans to easier traverse the indices.
	const auto add_conn_mutation_range = types::network_range_t::from_index_count(
		0,
		offspring_composition.add_conn_mutation_count
	);

	const auto add_node_mutation_range = types::network_range_t::from_index_count(
		add_conn_mutation_range.end(),
		offspring_composition.add_node_mutation_count
	);

	const auto conn_mutation_range = types::network_range_t::from_index_count(
		add_node_mutation_range.end(),
		offspring_composition.conn_mutation_count
	);

	const auto champion_range = types::network_range_t::from_index_count(
		conn_mutation_range.end(),
		offspring_composition.champion_count
	);

	const auto crossover_range = types::network_range_t::from_index_count(
		champion_range.end(),
		offspring_composition.crossover_count
	);

	// Extra ranges for the *algorithm*

	const auto directly_inherited_range = types::network_range_t::from_begin_end(
		add_conn_mutation_range.begin(),
		champion_range.end()
	);

	const auto topologically_unchanged_range = types::network_range_t::from_begin_end(
		conn_mutation_range.begin(),
		champion_range.end()
	);

	// Calculate the exact number of all offspring connections.
	types::population_composition_t connection_composition{};
	auto offspring_connection_count = std::size_t{};

	std::cout << "|-------------[ count add_conn_mutation connections ]-------------|" << std::endl;

	// Calculate number of add connection mutation networks
	for (const auto& network_index : add_conn_mutation_range.indices()) {

		const auto& ancestor_network = ancestors.networks[ancestor_lookup[network_index]];
		auto& offspring_network = offspring.networks[network_index];

		offspring_network.hidden_node_count = ancestor_network.hidden_node_count;

		auto& offspring_connections = offspring_network.connections;
		offspring_connections.begin() = offspring_connection_count + connection_composition.add_conn_mutation_count;
		offspring_connections.resize(ancestor_network.connections.size() + 1); // the inserted connection

		connection_composition.add_conn_mutation_count += offspring_connections.size();
	}
	offspring_connection_count += connection_composition.add_conn_mutation_count;

	std::cout << "|-------------[ count add_node_mutation connections ]-------------|" << std::endl;

	// Calculate number of add node mutation networks
	for (const auto& network_index : add_node_mutation_range.indices()) {

		const auto& ancestor_network = ancestors.networks[ancestor_lookup[network_index]];
		auto& offspring_network = offspring.networks[network_index];

		offspring_network.hidden_node_count = ancestor_network.hidden_node_count + 1;

		auto& offspring_connections = offspring_network.connections;
		offspring_connections.begin() = offspring_connection_count + connection_composition.add_node_mutation_count;
		offspring_connections.resize(
			ancestor_network.connections.size() + 2
		); // the two extra connection to and from the node

		connection_composition.add_node_mutation_count += offspring_connections.size();
	}

	offspring_connection_count += connection_composition.add_node_mutation_count;

	std::cout << "|-------------[ count topologically_unchanged connections ]-------------|" << std::endl;

	// Calculate number of topologically unchanged networks
	for (const auto& network_index : topologically_unchanged_range.indices()) {

		const auto& ancestor_network = ancestors.networks[ancestor_lookup[network_index]];
		auto& offspring_network = offspring.networks[network_index];

		offspring_network.hidden_node_count = ancestor_network.hidden_node_count;

		auto& offspring_connections = offspring_network.connections;
		offspring_connections.begin() = offspring_connection_count + connection_composition.conn_mutation_count;
		offspring_connections.resize(ancestor_network.connections.size()); // same topology -> no extra connections

		connection_composition.conn_mutation_count += offspring_connections.size();
	}

	offspring_connection_count += connection_composition.conn_mutation_count;

	std::cout << "|-------------[ count crossover connections ]-------------|" << std::endl;

	// Calculate number of crossover connections
	debug_vector<std::random_device::result_type> crossover_seeds(offspring_composition.crossover_count);

	const auto crossover_parent_lookup = debug_span(
		reinterpret_cast<const types::parents_t*>(&ancestor_lookup[crossover_range.begin()]),
		crossover_range.size()
	);
	assert(
		reinterpret_cast<const types::network_index_t*>(crossover_parent_lookup.end().base()) ==
		ancestor_lookup.cend().base()
	);

	std::random_device rd{};
	std::default_random_engine deterministic_engine{ 0 }; // This initial seed is just a placeholder and never used.
	auto parents_connection_survival_distrib = std::uniform_int_distribution<std::uint8_t>(false, true);

	for (types::network_index_t i{}; i != crossover_range.size(); ++i) {

		deterministic_engine.seed(crossover_seeds[i] = rd());

		parents_connection_survival_distrib.reset();

		auto& offspring_network = offspring.networks[crossover_range.begin() + i];

		auto& offspring_connections = offspring_network.connections;
		offspring_connections.begin() = offspring_connection_count + connection_composition.crossover_count;
		offspring_connections.resize(crossover_offspring_conn_count(
			deterministic_engine,
			parents_connection_survival_distrib,
			ancestors.networks,
			ancestors.connection_infos,
			ancestor_fitness,
			crossover_parent_lookup[i]
		));

		connection_composition.crossover_count += offspring_connections.size();
	}

	offspring_connection_count += connection_composition.crossover_count;

	// Finally allocate connection arrays
	offspring.connections.resize(offspring_connection_count);
	offspring.connection_weights.resize(offspring_connection_count);
	offspring.connection_infos.resize(offspring_connection_count);

	const auto calc_portion = [](const std::size_t count, const double portion) {
		return std::max(static_cast<std::size_t>(std::round(portion * static_cast<double>(count))), 1ul);
	};

	const auto calc_ratio_portion =
		[&calc_portion](const auto count, const std::size_t total, const std::size_t segment) {
			return calc_portion(count, static_cast<double>(total) / static_cast<double>(segment));
		};

	const auto mutation_tread_range = integer_range<unsigned int>::from_index_count(
		0,
		calc_portion(m_thread_count, 0.3)
	);

	const auto bytes_per_connection =
		(sizeof(types::connection_t) + sizeof(types::connection_weight_t) + sizeof(types::connection_info_t));

	const auto connection_copy_thread_count = calc_ratio_portion(
		mutation_tread_range.size(),
		bytes_per_connection,
		sizeof(types::connection_t)
	);
	const auto connection_weight_copy_thread_count = calc_ratio_portion(
		mutation_tread_range.size(),
		bytes_per_connection,
		sizeof(types::connection_weight_t)
	);
	const auto connection_info_copy_thread_count = calc_ratio_portion(
		mutation_tread_range.size(),
		bytes_per_connection,
		sizeof(types::connection_info_t)
	);

	std::cout << "|-------------[ copying directly inherited connections ]-------------|" << std::endl;

	// Copy old connections over from all directly inherited networks.
	if (not directly_inherited_range.empty()) {
		for (const auto directly_inherited_segment :
		     directly_inherited_range.balanced_segments(connection_copy_thread_count)) {
			threads.emplace_back([&, directly_inherited_segment]() {
				copy_connection_data<types::connection_t>(
					ancestor_lookup,
					ancestors.networks,
					offspring.networks,
					ancestors.connections,
					offspring.connections,
					directly_inherited_segment
				);
			});
		}

		for (const auto directly_inherited_segment :
		     directly_inherited_range.balanced_segments(connection_weight_copy_thread_count)) {
			threads.emplace_back([&, directly_inherited_segment]() {
				copy_connection_data<types::connection_weight_t>(
					ancestor_lookup,
					ancestors.networks,
					offspring.networks,
					ancestors.connection_weights,
					offspring.connection_weights,
					directly_inherited_segment
				);
			});
		}

		for (const auto directly_inherited_segment :
		     directly_inherited_range.balanced_segments(connection_info_copy_thread_count)) {
			threads.emplace_back([&, directly_inherited_segment]() {
				copy_connection_data<types::connection_info_t>(
					ancestor_lookup,
					ancestors.networks,
					offspring.networks,
					ancestors.connection_infos,
					offspring.connection_infos,
					directly_inherited_segment
				);
			});
		}
	}

	// Crossover copies the connections itself, so it can be started before the mutation copy finishes
	const auto crossover_thread_count = std::max(1u, m_thread_count - mutation_tread_range.size());

	std::cout << "|-------------[ starting crossocer ]-------------|" << std::endl;

	if (not crossover_range.empty()) {
		for (const auto& crossover_segment : crossover_range.balanced_segments(crossover_thread_count)) {
			threads.emplace_back([&, crossover_segment]() {
				create_crossovers(
					ancestors.networks,
					ancestors.connections,
					ancestors.connection_weights,
					ancestors.connection_infos,
					ancestor_fitness,
					crossover_parent_lookup,
					crossover_seeds,
					offspring.networks,
					offspring.connections,
					offspring.connection_weights,
					offspring.connection_infos,
					crossover_segment.begin() - crossover_range.begin(),
					crossover_segment
				);
			});
		}
	}

	// Wait for mutation copy threads to terminate
	const auto mutation_treads = mutation_tread_range.span<std::thread>(threads);
	for (auto& mutation_tread : mutation_treads) {
		mutation_tread.join();
	}
	threads.erase(threads.cbegin(), threads.cbegin() + mutation_treads.size());

	std::cout << "|-------------[ copying directly inherited connections done ]-------------|" << std::endl;

	// Start the mutation threads.
	const auto add_conn_mutation_thread_count = calc_portion(mutation_treads.size(), 0.6);
	const auto add_node_mutation_thread_count = calc_portion(mutation_treads.size(), 0.4);

	std::cout << "|-------------[ apply_add_conn_mutations ]-------------|" << std::endl;

	if (not add_conn_mutation_range.empty()) {
		for (const auto add_conn_mutation_segment :
		     add_conn_mutation_range.balanced_segments(add_conn_mutation_thread_count)) {
			threads.emplace_back([&, add_conn_mutation_segment]() {
				apply_add_conn_mutations(
					ancestors.networks,
					ancestors.connections,
					ancestor_lookup,
					offspring.networks,
					offspring.connections,
					offspring.connection_weights,
					offspring.connection_infos,
					add_conn_mutation_segment
				);
			});
		}
	}

	std::cout << "|-------------[ apply_add_node_mutations ]-------------|" << std::endl;

	if (not add_node_mutation_range.empty()) {
		for (const auto add_node_mutation_segment :
		     add_node_mutation_range.balanced_segments(add_node_mutation_thread_count)) {
			threads.emplace_back([&, add_node_mutation_segment]() {
				apply_add_node_mutations(
					ancestors.networks,
					ancestors.connections,
					ancestors.connection_weights,
					ancestor_lookup,
					offspring.networks,
					offspring.connections,
					offspring.connection_weights,
					offspring.connection_infos,
					add_node_mutation_segment
				);
			});
		}
	}

	// Wait for all topological mutations and crossovers to be done.
	for (auto& thread : threads) {
		thread.join();
	}
	threads.clear();

	// Apply simple connection weight mutations

	// Only current gen innovations need to be taken into account
	m_conn_lookup.clear();
	const auto all_conn_mutation_thread_count = calc_portion(m_thread_count, 0.2);

	std::cout << "|-------------[ mutate_connections ]-------------|" << std::endl;

	if (not conn_mutation_range.empty()) {
		for (const auto& conn_mutation_segment :
		     conn_mutation_range.balanced_segments(all_conn_mutation_thread_count)) {
			threads.emplace_back([&, conn_mutation_segment]() {
				mutate_all_connections(offspring.networks, offspring.connection_weights, conn_mutation_segment);
			});
		}
	}

	const auto some_conn_mutation_thread_count = calc_portion(m_thread_count, 0.8);
	const auto total_some_conn_mutations =
		(add_conn_mutation_range.size() + add_node_mutation_range.size() + crossover_range.size());

	if (not add_conn_mutation_range.empty()) {
		const auto add_conn_weight_mutation_thread_count = calc_ratio_portion(
			some_conn_mutation_thread_count,
			total_some_conn_mutations,
			add_conn_mutation_range.size()
		);

		for (const auto& conn_mutation_segment :
		     add_conn_mutation_range.balanced_segments(add_conn_weight_mutation_thread_count)) {
			threads.emplace_back([&, conn_mutation_segment]() {
				mutate_some_connections(offspring.networks, offspring.connection_weights, conn_mutation_segment);
			});
		}
	}

	if (not add_node_mutation_range.empty()) {
		const auto add_node_weight_mutation_thread_count = calc_ratio_portion(
			some_conn_mutation_thread_count,
			total_some_conn_mutations,
			add_node_mutation_range.size()
		);

		for (const auto& conn_mutation_segment :
		     add_node_mutation_range.balanced_segments(add_node_weight_mutation_thread_count)) {
			threads.emplace_back([&, conn_mutation_segment]() {
				mutate_some_connections(offspring.networks, offspring.connection_weights, add_node_mutation_range);
			});
		}
	}

	if (not crossover_range.empty()) {
		const auto crossover_weight_mutation_thread_count = calc_ratio_portion(
			some_conn_mutation_thread_count,
			total_some_conn_mutations,
			crossover_range.size()
		);

		for (const auto& conn_mutation_segment :
		     add_conn_mutation_range.balanced_segments(crossover_weight_mutation_thread_count)) {
			threads.emplace_back([&, conn_mutation_segment]() {
				mutate_some_connections(offspring.networks, offspring.connection_weights, conn_mutation_segment);
			});
		}
	}

	for (auto& thread : threads) {
		thread.join();
	}
	threads.clear();

	m_species_sorter.clear();

	for (const auto& network : add_conn_mutation_range.span<const types::network_t>(offspring.networks)) {
		for (const auto& [from, to] : network.connections.span<const types::connection_t>(offspring.connections)) {
			assert(from != to);
			assert(
				from < m_network_interface_config.input_count or
				(from >= (m_network_interface_config.input_count + m_network_interface_config.output_count) and
			     from < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			             network.hidden_node_count))
			);
			assert(
				to >= m_network_interface_config.input_count and
				to < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			          network.hidden_node_count)
			);
		}
	}

	for (const auto& network : add_node_mutation_range.span<const types::network_t>(offspring.networks)) {
		for (const auto& [from, to] : network.connections.span<const types::connection_t>(offspring.connections)) {
			// std::cout << from << "->" << to << std::endl;
			assert(from != to);
			assert(
				from < m_network_interface_config.input_count or
				(from >= (m_network_interface_config.input_count + m_network_interface_config.output_count) and
			     from < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			             network.hidden_node_count))
			);
			assert(
				to >= m_network_interface_config.input_count and
				to < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			          network.hidden_node_count)
			);
		}
	}

	for (const auto& network : conn_mutation_range.span<const types::network_t>(offspring.networks)) {
		for (const auto& [from, to] : network.connections.span<const types::connection_t>(offspring.connections)) {
			assert(from != to);
			assert(
				from < m_network_interface_config.input_count or
				(from >= (m_network_interface_config.input_count + m_network_interface_config.output_count) and
			     from < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			             network.hidden_node_count))
			);
			assert(
				to >= m_network_interface_config.input_count and
				to < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			          network.hidden_node_count)
			);
		}
	}

	for (const auto& network : champion_range.span<const types::network_t>(offspring.networks)) {
		for (const auto& [from, to] : network.connections.span<const types::connection_t>(offspring.connections)) {
			assert(from != to);
			assert(
				from < m_network_interface_config.input_count or
				(from >= (m_network_interface_config.input_count + m_network_interface_config.output_count) and
			     from < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			             network.hidden_node_count))
			);
			assert(
				to >= m_network_interface_config.input_count and
				to < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			          network.hidden_node_count)
			);
		}
	}

	for (const auto& network : crossover_range.span<const types::network_t>(offspring.networks)) {
		for (const auto& [from, to] : network.connections.span<const types::connection_t>(offspring.connections)) {
			assert(from != to);
			assert(
				from < m_network_interface_config.input_count or
				(from >= (m_network_interface_config.input_count + m_network_interface_config.output_count) and
			     from < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			             network.hidden_node_count))
			);
			assert(to >= m_network_interface_config.input_count);
			assert(
				to < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			          network.hidden_node_count)
			);
		}
	}

	std::cout << "|-------------[ sort_into_buckets ]-------------|" << std::endl;

	const auto offspring_network_range = types::network_range_t::from_range(offspring.networks);
	for (const auto& species_segment : offspring_network_range.balanced_segments(m_thread_count)) {
		threads.emplace_back([&, species_segment]() {
			m_species_sorter.sort_into_buckets(
				m_evolution_config.difference_config,
				offspring.connection_weights,
				offspring.connection_infos,
				offspring.networks,
				ancestors.networks.size() / ancestors.species.size(),
				species_segment
			);
		});
	}

	for (auto& thread : threads) {
		thread.join();
	}

	std::cout << "|-------------[ assign_species_and_sorted_networks ]-------------|" << std::endl;

	m_species_sorter.assign_species_and_sorted_networks(offspring.species, offspring.networks);

	std::cout << "Checking final connections" << std::endl;
	for (const auto& network : add_conn_mutation_range.span<const types::network_t>(offspring.networks)) {
		for (const auto& [from, to] : network.connections.span<const types::connection_t>(offspring.connections)) {
			assert(
				from < m_network_interface_config.input_count or
				(from >= (m_network_interface_config.input_count + m_network_interface_config.output_count) and
			     from < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			             network.hidden_node_count))
			);
			assert(
				to >= m_network_interface_config.input_count and
				to < (m_network_interface_config.input_count + m_network_interface_config.output_count +
			          network.hidden_node_count)
			);
		}
	}
}

void trainer::update_inference_network_group(inference::types::network_group_t& network_group) {
	auto& current_generation = m_populations[m_current_generation_index];

	auto num_inference_nodes = current_generation.networks.size() * m_network_interface_config.output_count;
	for (const auto& network : current_generation.networks) {
		num_inference_nodes += network.hidden_node_count;
	}

	network_group.nodes.resize(num_inference_nodes);
	network_group.connections.resize(current_generation.connections.size());
	network_group.networks.resize(current_generation.networks.size());

	debug_vector<std::thread> threads;
	threads.reserve(m_thread_count);

	const auto avg_conns_per_section = network_group.connections.size() / m_thread_count;

	auto network_section = types::network_range_t{};
	auto node_section = types::network_range_t{}; // TODO node range might be too small
	auto conn_section = types::conn_range_t{};

	auto section_conn_limit = avg_conns_per_section;

	while (network_section.end() != current_generation.networks.size()) {

		const auto& network = current_generation.networks[network_section.end()];
		node_section.end() += m_network_interface_config.output_count + network.hidden_node_count;
		conn_section.end() += network.connections.size();
		++network_section.end();

		if (conn_section.end() >= section_conn_limit) {

			threads.emplace_back(
				[this, &current_generation, &network_group, network_section, node_section, conn_section]() mutable {
					update_inference_network_section(
						current_generation,
						network_group,
						network_section,
						node_section,
						conn_section
					);
				}
			);

			network_section.begin() = network_section.end();
			node_section.begin() = node_section.end();
			conn_section.begin() = conn_section.end();

			section_conn_limit += avg_conns_per_section;
		}
	}

	for (auto& thread : threads) {
		thread.join();
	}
}

void trainer::update_inference_network_section(
	const types::population_t& generation,
	inference::types::network_group_t& network_group,
	const types::network_range_t& network_range,
	types::network_range_t& node_range,
	types::conn_range_t& conn_range
) {
	debug_vector<bool> node_visited;
	debug_vector<types::node_index_t> to_be_visited_nodes;
	std::vector<std::size_t> node_index_lookup;

	const auto output_range = types::conn_range_t::from_index_count(
		m_network_interface_config.input_count,
		m_network_interface_config.output_count
	);

	for (const auto& network_index : network_range.indices()) {

		const auto& network = generation.networks[network_index];
		auto& inference_network = network_group.networks[network_index];

		inference_network.nodes.begin() = node_range.begin();
		inference_network.nodes.clear();
		inference_network.connections_begin = conn_range.begin();

		const auto max_node_count = m_network_interface_config.output_count + network.hidden_node_count;
		to_be_visited_nodes.reserve(max_node_count);

		node_visited.clear();
		node_visited.resize(max_node_count, false);

		node_index_lookup.clear();
		node_index_lookup.resize(max_node_count, invalid_node_index);

		for (const auto& output_index : output_range.indices()) {
			to_be_visited_nodes.push_back(output_index);

			while (not to_be_visited_nodes.empty()) {

				const auto dst_node_index = to_be_visited_nodes.back();
				assert(dst_node_index >= m_network_interface_config.input_count);

				if (node_visited[dst_node_index - m_network_interface_config.input_count]) {
					to_be_visited_nodes.pop_back();
					continue;
				}

				auto all_incoming_visited = true;

				// First visit all child nodes, as their results need to be calculated first.
				for (const auto& conn_index : network.connections.indices()) {
					if (not generation.connection_infos[conn_index].enabled) {
						continue;
					}
					const auto& conn = generation.connections[conn_index];
					if (conn.to != dst_node_index) {
						continue;
					}

					if (conn.from >= m_network_interface_config.input_count and
					    node_visited[conn.from - m_network_interface_config.input_count]) {
						to_be_visited_nodes.push_back(conn.from);
						all_incoming_visited = false;
					}
				}

				if (not all_incoming_visited) {
					continue;
				}

				auto incoming_connection_count = inference::types::conn_index_t{};

				for (const auto& conn_index : network.connections.indices()) {
					if (not generation.connection_infos[conn_index].enabled) {
						continue;
					}

					const auto& conn = generation.connections[conn_index];
					if (conn.to != dst_node_index) {
						continue;
					}

					network_group.connections[conn_range.begin()++] = {
						.source_node_index = conn.from,
						.weight = generation.connection_weights[conn_index]
					};

					++incoming_connection_count;
				}

				const auto next_node_index = node_range.begin()++;
				auto& next_node = network_group.nodes[next_node_index];
				next_node.node_index = dst_node_index;
				next_node.incoming_connection_count = incoming_connection_count;
				++inference_network.nodes.end();

				node_index_lookup[dst_node_index - m_network_interface_config.input_count] = next_node_index;

				// Now that the node has been added, it can finally be removed from the stack.
				to_be_visited_nodes.pop_back();
			}
		}

		const auto remap_node = [&](std::size_t node_index) {
			if (node_index >= m_network_interface_config.input_count) {
				node_index = node_index_lookup[node_index - m_network_interface_config.input_count];
			}
			return static_cast<neat::inference::types::node_index_t>(node_index);
		};

		for (auto& node : inference_network.nodes.span<neat::inference::types::node_t>(network_group.nodes)) {
			node.node_index = remap_node(node.node_index);
		}
		for (auto& conn : inference_network.nodes.span<neat::inference::types::connection_t>(network_group.connections)) {
			conn.source_node_index = remap_node(conn.source_node_index);
		}
	}
}

bool trainer::would_create_loop(
	debug_span<const types::connection_t> connections,
	const types::network_t& network,
	const types::conn_index_t& src_index,
	const types::conn_index_t& dst_index,
	debug_vector<types::node_index_t>& node_stack
) const {

	// This function assumes that without the given connection the network is well-formed and loop free.

	const auto network_connections = network.connections.span(connections);

	node_stack.clear();
	node_stack.reserve(
		m_network_interface_config.input_count + m_network_interface_config.output_count + network.hidden_node_count
	);
	node_stack.push_back(src_index);

	std::size_t max_iterations = 100'000;

	while (not node_stack.empty()) {

		if (max_iterations-- == 0) {
			std::cerr << "would_create_loop called on network that already contains loop!" << std::endl;
			return true;
		}

		const auto node_index = node_stack.back();
		node_stack.pop_back();

		if (node_index < m_network_interface_config.input_count) {
			// It is not possible for a single connection to add a loop with an input node,
			// since no other connection can output to the input node.
			continue;
		}

		for (const auto& incoming_conn : network_connections) {
			if (incoming_conn.to == node_index) {
				if (incoming_conn.from == dst_index) {
					return true;
				} else {
					node_stack.push_back(incoming_conn.from);
				}
			}
		}
	}

	return false;
}

bool trainer::does_fitness_match(const types::fitness_t& a, const types::fitness_t& b) const {
	return std::abs(a - b) < m_evolution_config.fitness_epsilon;
}

} // namespace neat
