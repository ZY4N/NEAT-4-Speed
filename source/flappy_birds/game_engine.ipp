namespace flappy_birds {

template<class Renderer>
game_engine_t<Renderer>::game_engine_t(
	const game_logic::config_t& game_config,
	const renderer_config_t& renderer_config,
	const std::size_t bird_count,
	const int window_width,
	const int window_height
) :
	m_game_config(game_config), m_bird_count{ bird_count }, m_renderer{ renderer_config } {
	default_view(window_width, window_height);
	reset();
}

template<class Renderer>
game_logic::state_t& game_engine_t<Renderer>::state() {
	return m_game_state;
}

template<class Renderer>
bool game_engine_t<Renderer>::update(const float dt) {
	return m_physics_engine.update(m_game_config, m_game_state, m_will_flap, dt);
}

template<class Renderer>
void game_engine_t<Renderer>::flap(std::size_t index) {
	// TODO remap these indices using the active_bird_indices lookup
	m_will_flap[index] = true;
}

template<class Renderer>
void game_engine_t<Renderer>::render(sf::RenderWindow& window) {
	m_renderer.render(m_view_config, m_game_config, m_game_state, window);
}

template<class Renderer>
void game_engine_t<Renderer>::reset() {

	m_game_state.bird_states.resize(m_bird_count);
	std::fill(
		m_game_state.bird_states.begin(),
		m_game_state.bird_states.end(),
		game_logic::bird_state_t{ .position_y = std::lerp(m_game_config.floor_y, m_game_config.ceiling_y, 0.5f),
	                              .velocity_y = 0.0f }
	);

	m_game_state.active_bird_indices.resize(m_bird_count);
	std::iota(m_game_state.active_bird_indices.begin(), m_game_state.active_bird_indices.end(), 0);

	m_will_flap.clear();
	m_will_flap.resize(m_bird_count, false);

	m_game_state.scores.clear();
	m_game_state.scores.resize(m_bird_count, 0.0f);

	m_game_state.pipe_gaps_y.resize(m_game_config.pipes_behind_bird + m_game_config.pipes_in_front_of_bird);

	// TODO this is very very bad....
	std::uniform_real_distribution gap_distrib(
		m_game_config.floor_y + m_game_config.pipe_spacing_y / 2.0f,
		m_game_config.ceiling_y - m_game_config.pipe_spacing_y / 2.0f
	);
	std::generate(m_game_state.pipe_gaps_y.begin(), m_game_state.pipe_gaps_y.end(), [&]() {
		return gap_distrib(m_game_state.rng);
	});

	m_game_state.pipe_position_x = m_game_config.pipe_spacing_x / 2.0f;
	m_game_state.pipes_surpassed_count = 0;
}

template<class Renderer>
void game_engine_t<Renderer>::default_view(const int window_width, const int window_height) {
	const auto space_to_left = static_cast<float>(m_game_config.pipes_behind_bird) * m_game_config.pipe_spacing_x;

	// Needs -1 otherwise pipes are shown spawning in.
	const auto space_to_right = static_cast<float>(m_game_config.pipes_in_front_of_bird - 1) *
		m_game_config.pipe_spacing_x;

	const auto game_width = space_to_left + space_to_right;
	const auto game_height = 1.2f * (m_game_config.ceiling_y - m_game_config.floor_y);

	const auto scale_x = static_cast<float>(window_width) / game_width;
	const auto scale_y = static_cast<float>(window_height) / game_height;

	m_view_config.scale = std::min(scale_x, scale_y);

	m_view_config
		.center_x = std::lerp(m_game_config.bird_x - space_to_left, m_game_config.bird_x + space_to_right, 0.5f);

	m_view_config.center_y = std::lerp(m_game_config.floor_y, m_game_config.ceiling_y, 0.5f);
}

template<class Renderer>
std::span<const float> game_engine_t<Renderer>::scores() {
	return m_game_state.scores;
}

} // namespace flappy_birds
