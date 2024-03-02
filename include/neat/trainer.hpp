#pragma once

#include "evolution_config.hpp"
#include "helpers/connection_lookup.hpp"
#include "helpers/species_sorter.hpp"
#include "inference.hpp"
#include "network_interface_config.hpp"

#include <array>
#include <random>
#include <span>
#include "util/debug_span.hpp" // TODO remove
#include <system_error>
#include "util/debug_vector.hpp" // TODO remove

namespace neat {

class trainer {
	using seed_t = std::random_device::result_type;

public:
	trainer(
		const evolution_config_t& evolution_config,
		const network_interface_config_t& network_interface_config,
		std::size_t population_size,
		std::uint32_t thread_count
	);


	void evolve(debug_span<const types::fitness_t> ancestor_fitness, inference::types::network_group_t& network_group);

protected:
	void create_initial_population();

	void swap_population();

	void evolve_into(
		const types::population_t& ancestors,
		debug_span<const types::fitness_t> ancestor_fitness,
		types::population_t& offspring
	);

	void update_inference_network_group(inference::types::network_group_t& network_group);

private:
	bool would_create_loop(
		debug_span<const types::connection_t> connections,
		const types::network_t& network,
		const types::conn_index_t& src_index,
		const types::conn_index_t& dst_index,
		debug_vector<types::node_index_t>& node_stack
	) const;

	static void calc_species_fitness(
		debug_span<const types::species_t> all_species,
		debug_span<const types::fitness_t> network_fitness,
		debug_span<float> species_fitness,
		const types::species_range_t& species_range
	);

	void divide_offspring_between_species(
		debug_span<const float> ancestor_species_fitness, debug_vector<types::network_index_t>& species_offspring_counts
	);

	void calculate_species_offspring_composition_and_sample_ancestors(
		debug_span<const types::species_t> all_ancestor_species,
		debug_span<const types::network_t> all_ancestor_networks,
		debug_span<const float> ancestor_species_fitness,
		debug_span<const types::network_index_t> species_offspring_counts,
		debug_span<types::population_composition_t> offspring_species_composition,
		debug_span<debug_vector<types::network_index_t>> species_ancestor_lookups,
		const types::species_range_t& species_range
	);

	void apply_add_conn_mutations(
		debug_span<const types::network_t> ancestor_networks,
		debug_span<const types::connection_t> ancestor_connections,
		debug_span<const types::network_index_t> ancestor_lookup,
		debug_span<types::network_t> offspring_networks,
		debug_span<types::connection_t> offspring_connections,
		debug_span<types::connection_weight_t> offspring_connection_weights,
		debug_span<types::connection_info_t> offspring_connection_infos,
		const types::network_range_t& add_conn_mutation_range
	);

	void apply_add_node_mutations(
		debug_span<const types::network_t> ancestor_networks,
		debug_span<const types::connection_t> ancestor_connections,
		debug_span<const types::connection_weight_t> ancestor_connection_weights,
		debug_span<const types::network_index_t> ancestor_lookup,
		debug_span<types::network_t> offspring_networks,
		debug_span<types::connection_t> offspring_connections,
		debug_span<types::connection_weight_t> offspring_connection_weights,
		debug_span<types::connection_info_t> offspring_connection_infos,
		const types::network_range_t& add_node_mutation_range
	);

	void mutate_all_connections(
		debug_span<const types::network_t> offspring_networks,
		debug_span<types::connection_weight_t> offspring_connection_weights,
		const types::network_range_t& conn_mutation_range
	);

	void mutate_some_connections(
		debug_span<const types::network_t> offspring_networks,
		debug_span<types::connection_weight_t> offspring_connection_weights,
		const types::network_range_t& conn_mutation_range
	);

	void create_crossovers(
		debug_span<const types::network_t> ancestor_networks,
		debug_span<const types::connection_t> ancestor_connections,
		debug_span<const types::connection_weight_t> ancestor_connection_weights,
		debug_span<const types::connection_info_t> ancestor_connection_infos,
		debug_span<const types::fitness_t> ancestor_fitness,
		debug_span<const types::parents_t> parents_lookup,
		debug_span<const seed_t> disjoint_excess_conn_selection_seeds,
		debug_span<types::network_t> offspring_networks,
		debug_span<types::connection_t> offspring_connections,
		debug_span<types::connection_weight_t> offspring_connection_weights,
		debug_span<types::connection_info_t> offspring_connection_infos,
		std::size_t seed_offset,
		const types::network_range_t& crossover_range
	);

	std::size_t crossover_offspring_conn_count(
		std::default_random_engine& disjoint_and_excess_connection_selection_engine,
		std::uniform_int_distribution<std::uint8_t>& parents_connection_survival_distrib,
		debug_span<const types::network_t> ancestor_networks,
		debug_span<const types::connection_info_t> ancestor_connection_infos,
		debug_span<const types::fitness_t> ancestor_fitness,
		const types::parents_t& parent_indices
	);

	void update_inference_network_section(
		const types::population_t& generation,
		inference::types::network_group_t& network_group,
		const types::network_range_t& network_range,
		types::conn_range_t& node_range,
		types::conn_range_t& conn_range
	);

	[[nodiscard]] bool does_fitness_match(const types::fitness_t& a, const types::fitness_t& b) const;

private:
	evolution_config_t m_evolution_config;
	network_interface_config_t m_network_interface_config;
	std::size_t m_population_size;
	std::uint32_t m_thread_count;

	std::array<types::population_t, 2> m_populations;
	std::size_t m_current_generation_index{};

	std::default_random_engine m_rng;
	connection_lookup m_conn_lookup;
	species_sorter m_species_sorter;
};

} // namespace neat
