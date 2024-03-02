
#include "flappy_birds/game_engine.hpp"
#include "neat/trainer.hpp"

#include <SFML/Window/Event.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

int main() {

	auto res_width = 1'920, res_height = 1'080;

	const auto fps = 60;
	using seconds_t = std::chrono::duration<float>;
	const auto frame_time = seconds_t{ 1.0 } / static_cast<float>(fps);
	const auto dt = std::chrono::duration_cast<seconds_t>(frame_time).count(); // 2.0f;
	const auto stop_score = 100;

	// Network Inputs
	static constexpr auto dist_gap_y_index{ 0 }, dist_pipe_x_index{ 1 }, dist_to_ceiling_y{ 2 }, dist_to_floor_y{ 2 },
		bias_index{ 3 };

	const auto evolution_config = neat::evolution_config_t{};
	const auto interface_config = neat::network_interface_config_t{ .input_count = 5, .output_count = 1 };

	const auto population_size = 10'000;
	const auto thread_count = std::thread::hardware_concurrency();

	neat::trainer flappy_trainer(evolution_config, interface_config, population_size, thread_count);
	neat::inference::types::network_group_t inference_networks;

	debug_vector<float> inputs(population_size * interface_config.input_count);
	debug_vector<float> outputs(population_size * interface_config.output_count);

	debug_vector<neat::types::fitness_t> fitness(population_size, 0.0f);
	debug_vector<float> results(population_size);

	const auto game_config = flappy_birds::game_logic::config_t{};

	auto game_engine = flappy_birds::game_engine_t<flappy_birds::rendering::texture_renderer_t>(
		game_config,
		flappy_birds::rendering::texture_config_t{},
		population_size,
		res_width,
		res_height
	);

	const auto game_batch_size = 10;
	const auto batch_scale = 1.0f / static_cast<float>(game_batch_size);

	std::atomic_flag stop_training = ATOMIC_FLAG_INIT;

	auto keyboard_listener_thread = std::thread([&stop_training]() {
		std::cout << "Press [ENTER] to stop training." << std::endl;
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		stop_training.test_and_set(std::memory_order_acquire);
		std::cout << "Training will be stopped after the next generation." << std::endl;
	});

	const auto do_inference = [&]() {
		static debug_vector<std::thread> threads;
		threads.reserve(thread_count);
		auto& game_state = game_engine.state();

		const auto inference_network_range = integer_range<neat::types::network_index_t>::from_range(
			inference_networks.networks
		);

		const auto next_gap_y = game_state.pipe_gaps_y[game_config.pipes_behind_bird];

		for (std::size_t i{}; i != game_state.bird_states.size(); ++i) {

			const auto& bird_state = game_state.bird_states[i];

			const auto bird_index = game_state.active_bird_indices[i];
			const auto bird_inputs = inputs.begin() + bird_index * interface_config.input_count;

			bird_inputs[dist_gap_y_index] = next_gap_y - bird_state.position_y;

			bird_inputs[dist_pipe_x_index] = game_state.pipe_position_x;

			bird_inputs[dist_to_ceiling_y] = game_config.ceiling_y -
				(bird_state.position_y + game_config.bird_radius);

			bird_inputs[dist_to_floor_y] = (bird_state.position_y - game_config.bird_radius) - game_config.floor_y;

			bird_inputs[bias_index] = 1.0f;
		}

		for (const auto& inference_segment : inference_network_range.balanced_segments(thread_count)) {
			threads.emplace_back([&, inference_segment]() {
				neat::inference::evaluate_network_range(inference_networks, inputs, outputs, inference_segment);
			});
		}
		for (auto& thread : threads) {
			thread.join();
		}
		threads.clear();

		for (std::size_t i{}; i != game_state.active_bird_indices.size(); ++i) {
			if (outputs[game_state.active_bird_indices[i]] > 0.5f) {
				game_engine.flap(i);
			}
		}
	};

	auto generation_index = std::size_t{};
	while (not stop_training.test(std::memory_order_acquire)) {
		std::cout << "|--------[ generation " << generation_index << " ]--------|" << std::endl;

		flappy_trainer.evolve(fitness, inference_networks);

		std::cout << "Evaluating performance..." << std::endl;

		std::fill(fitness.begin(), fitness.end(), 0.0f);

		for (int i{}; i != game_batch_size; ++i) {
			game_engine.reset();

			do {
				do_inference();
			} while (not game_engine.update(dt));

			auto& game_state = game_engine.state();

			for (std::size_t j{}; j != fitness.size(); ++j) {
				const auto& bird_score = game_state.scores[j];
				if (bird_score >= stop_score) {
					stop_training.test_and_set(std::memory_order_acquire);
					break;
				}
				fitness[j] += batch_scale * game_state.scores[j];
			}
		}

		const auto [min_fitness_it, max_fitness_it] = std::minmax_element(fitness.begin(), fitness.end());
		std::cout << "Average scores min: " << *min_fitness_it << " max: " << *max_fitness_it << std::endl;
		++generation_index;
	}

	// keyboard_listener_thread.join();

	//----------------------[ Window/GLEW Setup ]----------------------//

	auto window = sf::RenderWindow(
		sf::VideoMode(res_width, res_height),
		"NEAT-4-Speed",
		sf::Style::Default,
		sf::ContextSettings(24, 8, 2, 4, 6)
	);
	auto [width, height] = window.getSize();

	bool running = true;
	bool game_over = false;
	bool pause = false;

	//----------------------[ Game Loop ]----------------------//

	while (running) {
		const auto start = std::chrono::high_resolution_clock::now();

		if (game_over) {
			std::cout << "score: " << game_engine.scores().front() << std::endl;
			game_engine.reset();
		}

		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				running = false;
			} else if (event.type == sf::Event::Resized) {
				window.setView(sf::View(
					{ 0.0f, 0.0f, static_cast<float>(event.size.width), static_cast<float>(event.size.height) }
				));
				width = event.size.width;
				height = event.size.height;
			} else if (event.type == sf::Event::MouseWheelMoved) {
				auto& scale = game_engine.m_view_config.scale;
				scale = std::max(scale * (1.0f + 0.1f * event.mouseWheel.delta), 0.1f);
			} else if (event.type == sf::Event::KeyPressed) {
				switch (event.key.code) {
				case sf::Keyboard::Escape:
					running = false;
					break;
				case sf::Keyboard::Tab:
					pause = !pause;
					break;
				case sf::Keyboard::R:
					game_engine.default_view(width, height);
					break;
				case sf::Keyboard::Space:
					game_engine.flap(0);
					break;
				default:
					break;
				}
			}
		}

		if (not pause) {
			do_inference();
			game_over = game_engine.update(dt);
		}

		game_engine.render(window);
		window.display();

		const auto finish = std::chrono::high_resolution_clock::now();
		std::this_thread::sleep_for(frame_time - (finish - start));
	}

	return EXIT_SUCCESS;
}

void xor_test() {

	// XOR lets go brrrr!!!

	const auto evolution_config = neat::evolution_config_t{};
	const auto interface_config = neat::network_interface_config_t{ .input_count = 3, // two inputs plus bias
		                                                            .output_count = 1 };

	const auto population_size = 1'000;
	const auto thread_count = 1; // std::thread::hardware_concurrency();

	neat::trainer xor_trainer(evolution_config, interface_config, population_size, thread_count);
	neat::inference::types::network_group_t network_group;
	std::array<float, 3> inputs;
	std::array<float, 1> outputs;
	debug_vector<neat::types::fitness_t> fitness(population_size, 0.0f);
	debug_vector<float> results(population_size);

	debug_vector<std::thread> threads;
	threads.reserve(thread_count);

	const auto to_float = [](const bool b) { return static_cast<float>(b); };
	const auto network_segments = integer_range<neat::types::network_index_t>::from_index_count(0, population_size);

	while (true) {
		xor_trainer.evolve(fitness, network_group);

		for (const auto& network_segment : network_segments.balanced_segments(thread_count)) {
			threads.emplace_back([&, network_segment]() {
				for (const auto& [a, b] : { std::pair(false, false),
				                            std::pair(false, true),
				                            std::pair(true, false),
				                            std::pair(true, true) }) {

					inputs = { to_float(a), to_float(b), 1.0f };
					outputs = { to_float(a != b) };

					neat::inference::evaluate_network_range(network_group, inputs, results, network_segment);
					for (const auto& i : network_segment.indices()) {
						fitness[i] -= std::pow(outputs[0] - results[i], 2);
					}
				}
			});
		}

		for (auto& thread : threads) {
			thread.join();
		}
		threads.clear();

		const auto [min_after, max_after] = std::minmax_element(fitness.begin(), fitness.end());

		std::cout << "Final Fitness: " << *min_after << " " << *max_after << std::endl;
	}
}
