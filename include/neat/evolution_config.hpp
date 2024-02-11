#pragma once

#include <cinttypes>

namespace neat {

struct mutation_rate_config_t {
	float new_connection_rate{ 0.05f };
	float new_node_rate{ 0.03f };
	float inter_species_mating_rate{ 0.001f };
	float keep_disabled_rate{ 0.75f };
	float offspring_mutation_rate{ 0.25f };
	float network_mutation_rate{ 0.8f };
	float uniform_mutation_rate{ 0.9f };
	std::uint32_t min_network_champion_size{ 5 };
};

struct extinction_config_t {
	float max_remove_population_portion{ 0.2 };
	float max_remove_score_portion{ 0.2 };
};

struct weight_distribution_config_t {
	float conn_weight_min{ 0.0f }, conn_weight_max{ 1.0f };
	float conn_weight_offset_min{ -0.01f }, conn_weight_offset_max{ 0.01f };
};

struct difference_config_t {
	float difference_threshold{ 3.0f };
	float difference_excess_weight{ 1.0f };
	float difference_disjoint_weight{ 1.0f };
	float difference_avg_weight_weights{ 0.4f };
};

struct evolution_config_t {
	difference_config_t difference_config;
	mutation_rate_config_t mutation_rate_config;
	extinction_config_t extinction_config;
	weight_distribution_config_t weight_distribution_config;
	float fitness_epsilon{ 0.001f };
};

} // namespace neat
