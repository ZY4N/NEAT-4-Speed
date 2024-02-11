#pragma once

#include "util/integer_range.hpp"

#include <cinttypes>
#include <limits>
#include <span>
#include "util/debug_span.hpp" // TODO remove
#include <vector>
#include "util/debug_vector.hpp" // TODO remove

namespace neat {

namespace types {

using innovation_number_t = std::uint64_t;
using connection_weight_t = float;
using fitness_t = float;
using network_difference_t = float;
using species_index_t = std::uint32_t;
using network_index_t = std::uint32_t;
using node_index_t = std::size_t;
using conn_index_t = std::size_t;
using parents_t = std::array<network_index_t, 2>;

using species_range_t = integer_range<species_index_t>;
using network_range_t = integer_range<network_index_t>;
using node_range_t = integer_range<node_index_t>;
using conn_range_t = integer_range<conn_index_t>;

struct population_composition_t {
	network_index_t add_conn_mutation_count{};
	network_index_t add_node_mutation_count{};
	network_index_t conn_mutation_count{};
	network_index_t champion_count{};
	network_index_t crossover_count{};
};

struct connection_t {
	node_index_t from, to;
};

struct connection_info_t {
	bool enabled : 1 { 0 };
	innovation_number_t innovation_number : (sizeof(innovation_number_t) * 8 - 1){ 0 };
};

struct network_t {
	node_index_t hidden_node_count;
	conn_range_t connections;
};

struct species_t {
	network_range_t networks;
};

struct population_t {
	debug_vector<species_t> species;
	debug_vector<network_t> networks;
	debug_vector<connection_t> connections;
	debug_vector<connection_weight_t> connection_weights;
	debug_vector<connection_info_t> connection_infos;
};
} // namespace types

constexpr inline auto invalid_species_index = std::numeric_limits<types::species_index_t>::max();
constexpr inline auto invalid_node_index = std::numeric_limits<types::node_index_t>::max();
constexpr inline auto invalid_conn_index = std::numeric_limits<types::conn_index_t>::max();
constexpr inline auto invalid_innovation_number = std::numeric_limits<types::innovation_number_t>::max();

} // namespace neat
