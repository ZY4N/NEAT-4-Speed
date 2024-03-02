#include <cinttypes>
#include <ranges>
#include <span>
#include "util/debug_span.hpp" // TODO remove
#include "util/debug_vector.hpp" // TODO remove
#include <cassert>

template<typename Integer>
class balanced_segments_t;

template<typename Integer>
class fixed_segments_t;

template<typename Integer>
struct integer_range {

	integer_range() = default;

	template<typename T>
	[[nodiscard]] inline static integer_range from_range(const T& range);

	[[nodiscard]] inline static integer_range from_begin_end(Integer begin, Integer end);

	[[nodiscard]] inline static integer_range from_index_count(Integer index, Integer count);

	[[nodiscard]] inline const Integer& begin() const;

	[[nodiscard]] inline const Integer& end() const;

	[[nodiscard]] inline Integer& begin();

	[[nodiscard]] inline Integer& end();

	[[nodiscard]] inline Integer size() const;

	inline void resize(const Integer& new_size);

	inline void clear();

	[[nodiscard]] inline bool empty() const;

	[[nodiscard]] inline bool contains(const Integer& index) const;

	[[nodiscard]] inline auto indices() const;

	template<typename T>
	[[nodiscard]] inline debug_span<T> span(debug_span<T> range) const;

	template<typename T>
	[[nodiscard]] inline debug_span<const T> span(const debug_vector<T>& range) const;

	template<typename T>
	[[nodiscard]] inline debug_span<T> span(debug_vector<T>& range) const;


	template<typename T>
	[[nodiscard]] inline debug_span<const T> cspan(debug_span<T> range) const;

	template<typename T>
	[[nodiscard]] inline debug_span<const T> cspan(const debug_vector<T>& range) const;

	[[nodiscard]] inline balanced_segments_t<Integer> balanced_segments(std::size_t segment_count) const;
	[[nodiscard]] inline fixed_segments_t<Integer> fixed_segments(Integer segment_size) const;

private:
	integer_range(Integer begin, Integer end);

	Integer m_begin{}, m_end{};
};


template<typename Integer>
class balanced_segment_iterator_t {
public:
	using value_type = integer_range<Integer>;
	using size_type = std::uint64_t;

public:
	[[nodiscard]] inline static balanced_segment_iterator_t make_begin(
		const value_type& range,const size_type& num_segments
	);

	[[nodiscard]] inline static balanced_segment_iterator_t make_end(
		const value_type& range, const size_type& num_segments
	);

	inline balanced_segment_iterator_t& operator++();
	[[nodiscard]] inline balanced_segment_iterator_t operator+(size_type offset) const;
	[[nodiscard]] inline value_type operator*() const;
	[[nodiscard]] inline value_type operator[](size_type offset) const;

	[[nodiscard]] inline bool operator==(const balanced_segment_iterator_t&) const;
	[[nodiscard]] inline bool operator!=(const balanced_segment_iterator_t&) const;

private:
	inline balanced_segment_iterator_t(
		const value_type& range, const size_type& num_segments, const size_type& segment_index
	);

private:
	const Integer m_begin, m_min_segment_size, m_remaining_values;
	size_type m_segment_index;
};

template<typename Integer>
class balanced_segments_t {
public:
	using iterator = balanced_segment_iterator_t<Integer>;
	using value_type = typename iterator::value_type;
	using size_type = typename iterator::size_type;

public:
	inline balanced_segments_t(const integer_range<Integer>& range, const size_type& num_segments);

	[[nodiscard]] inline const iterator& begin() const;
	[[nodiscard]] inline const iterator& end() const;

	value_type operator[](std::size_t index) const;

private:
	iterator m_begin, m_end;
};


template<typename Integer>
class fixed_segment_iterator_t {
public:
	using value_type = integer_range<Integer>;
	using size_type = std::uint64_t;

public:
	[[nodiscard]] inline static fixed_segment_iterator_t make_begin(const integer_range<Integer>& range, const Integer& segment_size);
	[[nodiscard]] inline static fixed_segment_iterator_t make_end(const integer_range<Integer>& range, const Integer& segment_size);

	inline fixed_segment_iterator_t& operator++();
	[[nodiscard]] inline fixed_segment_iterator_t operator+(size_type offset) const;
	[[nodiscard]] inline value_type operator*() const;
	[[nodiscard]] inline value_type operator[](size_type offset) const;

	[[nodiscard]] inline bool operator==(const fixed_segment_iterator_t&) const;
	[[nodiscard]] inline bool operator!=(const fixed_segment_iterator_t&) const;

private:
	inline fixed_segment_iterator_t(
		const integer_range<Integer>& full_range, const Integer& segment_size, const size_type& segment_index
	);

	const value_type m_full_range;
	const Integer m_segment_size;
	size_type m_segment_index;
};

template<typename Integer>
class fixed_segments_t {
public:
	using iterator = fixed_segment_iterator_t<Integer>;
	using value_type = typename iterator::value_type;
	using size_type = typename iterator::size_type;

	inline fixed_segments_t(const integer_range<Integer>& range, const Integer& segment_size);

	[[nodiscard]] inline const iterator& begin() const;
	[[nodiscard]] inline const iterator& end() const;

	[[nodiscard]] inline value_type operator[](size_type index) const;

private:
	fixed_segment_iterator_t<Integer> m_begin, m_end;
};


//--------------------[ balanced_segment_iterator_t ]--------------------//

template<class Integer>
balanced_segment_iterator_t<Integer>::balanced_segment_iterator_t(
	const integer_range<Integer>& range, const size_type& num_segments, const size_type& segment_index
) :
	m_begin{ range.begin() },
	m_min_segment_size{ range.size() / static_cast<Integer>(num_segments) },
	m_remaining_values{ range.size() % static_cast<Integer>(num_segments) },
	m_segment_index{ segment_index } {
}

template<class Integer>
balanced_segment_iterator_t<Integer> balanced_segment_iterator_t<Integer>::make_begin(
	const integer_range<Integer>& range, const size_type& num_segments
) {
	return { range, num_segments, 0 };
}

template<class Integer>
balanced_segment_iterator_t<Integer> balanced_segment_iterator_t<Integer>::make_end(
	const integer_range<Integer>& range, const size_type& num_segments
) {
	return { range, num_segments, num_segments };
}

template<class Integer>
typename balanced_segment_iterator_t<Integer>::value_type balanced_segment_iterator_t<Integer>::operator*() const {

	const auto index_integer = static_cast<Integer>(m_segment_index);

	const auto segment_begin = m_begin + m_segment_index * m_min_segment_size +
		std::min(static_cast<Integer>(m_segment_index), m_remaining_values);
	const auto segment_size = m_min_segment_size + static_cast<Integer>(size_type{ index_integer < m_remaining_values });

	return value_type::from_index_count(segment_begin, segment_size);
}


template<class Integer>
balanced_segment_iterator_t<Integer>& balanced_segment_iterator_t<Integer>::operator++() {
	++m_segment_index;
	return *this;
}

template<class Integer>
balanced_segment_iterator_t<Integer> balanced_segment_iterator_t<Integer>::operator+(const size_type offset) const {
	auto copy = *this;
	copy.m_segment_index += offset;
	return copy;
}

template<class Integer>
typename balanced_segment_iterator_t<Integer>::value_type balanced_segment_iterator_t<Integer>::operator[](
	const size_type offset
) const {
	return *(*this + offset);
}

template<class Integer>
bool balanced_segment_iterator_t<Integer>::operator==(const balanced_segment_iterator_t& other) const {
	return this->m_segment_index == other.m_segment_index and this->m_begin == other.m_begin and
		this->m_min_segment_size == other.m_min_segment_size and this->m_remaining_values == other.m_remaining_values;
}

template<class Integer>
bool balanced_segment_iterator_t<Integer>::operator!=(const balanced_segment_iterator_t& other) const {
	return not(*this == other);
}


//--------------------[ balanced_segments_t ]--------------------//

template<class Integer>
balanced_segments_t<Integer>::balanced_segments_t(const integer_range<Integer>& range, const size_type& num_segments) :
	m_begin{ balanced_segment_iterator_t<Integer>::make_begin(range, num_segments) },
	m_end{ balanced_segment_iterator_t<Integer>::make_end(range, num_segments) } {
}

template<class Integer>
const typename balanced_segments_t<Integer>::iterator& balanced_segments_t<Integer>::begin() const {
	return m_begin;
}

template<class Integer>
const typename balanced_segments_t<Integer>::iterator& balanced_segments_t<Integer>::end() const {
	return m_end;
}

template<class Integer>
typename balanced_segments_t<Integer>::value_type balanced_segments_t<Integer>::operator[](std::size_t index) const {
	return m_begin[index];
}


//--------------------[ fixed_segment_iterator_t ]--------------------//

template<class Integer>
fixed_segment_iterator_t<Integer>::fixed_segment_iterator_t(
	const integer_range<Integer>& full_range, const Integer& segment_size, const size_type& segment_index
) :
	m_full_range{ full_range }, m_segment_size{ segment_size }, m_segment_index{ segment_index } {
}

template<class Integer>
fixed_segment_iterator_t<Integer> fixed_segment_iterator_t<Integer>::make_begin(
	const integer_range<Integer>& range, const Integer& segment_size
) {
	return { range, segment_size, 0 };
}

template<class Integer>
fixed_segment_iterator_t<Integer> fixed_segment_iterator_t<Integer>::make_end(
	const integer_range<Integer>& range, const Integer& segment_size
) {
	const auto num_segments = (range.size() + segment_size - Integer{ 1 }) / segment_size;
	return { range, segment_size, num_segments };
}

template<class Integer>
typename fixed_segment_iterator_t<Integer>::value_type fixed_segment_iterator_t<Integer>::operator*() const {

	const auto relative_begin = m_segment_size * m_segment_index;
	const auto segment_begin = m_full_range.begin() + relative_begin;
	const auto segment_end = std::min(segment_begin + m_segment_size, m_full_range.end());

	return value_type::from_begin_end(segment_begin, segment_end);
}

template<class Integer>
fixed_segment_iterator_t<Integer>& fixed_segment_iterator_t<Integer>::operator++() {
	++m_segment_index;
	return *this;
}

template<class Integer>
fixed_segment_iterator_t<Integer> fixed_segment_iterator_t<Integer>::operator+(const size_type offset) const {
	auto copy = *this;
	copy.m_segment_index += offset;
	return copy;
}

template<class Integer>
typename fixed_segment_iterator_t<Integer>::value_type fixed_segment_iterator_t<Integer>::operator[](const size_type offset
) const {
	return *(*this + offset);
}

template<class Integer>
bool fixed_segment_iterator_t<Integer>::operator==(const fixed_segment_iterator_t& other) const {
	return this->m_segment_index == other.m_segment_index and this->m_full_range == other.m_full_range and
		this->m_segment_size == other.m_segment_size;
}

template<class Integer>
bool fixed_segment_iterator_t<Integer>::operator!=(const fixed_segment_iterator_t& other) const {
	return not(*this == other);
}


//--------------------[ fixed_segments_t ]--------------------//

template<class Integer>
fixed_segments_t<Integer>::fixed_segments_t(const integer_range<Integer>& range, const Integer& segment_size) :
	m_begin{ fixed_segment_iterator_t<Integer>::make_begin(range, segment_size) },
	m_end{ fixed_segment_iterator_t<Integer>::make_end(range, segment_size) } {
}

template<class Integer>
const typename fixed_segments_t<Integer>::iterator& fixed_segments_t<Integer>::begin() const {
	return m_begin;
}

template<class Integer>
const typename fixed_segments_t<Integer>::iterator& fixed_segments_t<Integer>::end() const {
	return m_end;
}

template<class Integer>
typename fixed_segments_t<Integer>::value_type fixed_segments_t<Integer>::operator[](const size_type index) const {
	return m_begin[index];
}


//--------------------[ integer_range ]--------------------//

template<typename Integer>
template<typename T>
integer_range<Integer> integer_range<Integer>::from_range(const T& range) {
	return integer_range(0, std::end(range) - std::begin(range));
}

template<typename Integer>
integer_range<Integer> integer_range<Integer>::from_begin_end(const Integer begin, const Integer end) {
	return integer_range(begin, end);
}

template<typename Integer>
integer_range<Integer> integer_range<Integer>::from_index_count(const Integer index, const Integer count) {
	return integer_range(index, index + count);
}

template<typename Integer>
[[nodiscard]] const Integer& integer_range<Integer>::begin() const {
	return m_begin;
}

template<typename Integer>
[[nodiscard]] const Integer& integer_range<Integer>::end() const {
	return m_end;
}

template<typename Integer>
[[nodiscard]] Integer& integer_range<Integer>::begin() {
	return m_begin;
}

template<typename Integer>
[[nodiscard]] Integer& integer_range<Integer>::end() {
	return m_end;
}

template<typename Integer>
[[nodiscard]] Integer integer_range<Integer>::size() const {
	return m_end - m_begin;
}

template<typename Integer>
void integer_range<Integer>::resize(const Integer& new_size) {
	m_end = m_begin + new_size;
}

template<typename Integer>
void integer_range<Integer>::clear() {
	m_end = m_begin;
}

template<typename Integer>
[[nodiscard]] bool integer_range<Integer>::empty() const {
	return m_begin == m_end;
}

template<typename Integer>
[[nodiscard]] bool integer_range<Integer>::contains(const Integer& index) const {
	return m_begin <= index and index < m_end;
}

template<typename Integer>
[[nodiscard]] auto integer_range<Integer>::indices() const {
	return std::ranges::iota_view{ m_begin, m_end };
}

template<typename Integer>
[[nodiscard]] balanced_segments_t<Integer> integer_range<Integer>::balanced_segments(const std::size_t segment_count) const {
	return balanced_segments_t<Integer>(*this, segment_count);
}


template<typename Integer>
[[nodiscard]] fixed_segments_t<Integer> integer_range<Integer>::fixed_segments(const Integer segment_size) const {
	return fixed_segments_t<Integer>(*this, segment_size);
}

template<typename Integer>
template<typename T>
[[nodiscard]] debug_span<T> integer_range<Integer>::span(debug_span<T> range) const {
	assert(m_begin == m_end or m_begin < range.size());
	assert(m_end <= range.size());
	// return range.subspan(m_begin, m_end - m_begin); TODO use again
	return debug_span<T>(range.begin() + m_begin, range.begin() + m_end);
}

template<typename Integer>
template<typename T>
[[nodiscard]] debug_span<const T> integer_range<Integer>::span(const debug_vector<T>& range) const {
	assert(m_begin == m_end or m_begin < range.size());
	assert(m_end <= range.size());
	return debug_span<const T>(range.begin() + m_begin, range.begin() + m_end);
}

template<typename Integer>
template<typename T>
[[nodiscard]] debug_span<T> integer_range<Integer>::span(debug_vector<T>& range) const {
	assert(m_begin == m_end or m_begin < range.size());
	assert(m_end <= range.size());
	return debug_span<T>(range.begin() + m_begin, range.begin() + m_end);
}

template<typename Integer>
template<typename T>
[[nodiscard]] debug_span<const T> integer_range<Integer>::cspan(debug_span<T> range) const {
	assert(m_begin == m_end or m_begin < range.size());
	assert(m_end <= range.size());
	return debug_span<const T>(range.begin() + m_begin, range.begin() + m_end);
}

template<typename Integer>
template<typename T>
[[nodiscard]] debug_span<const T> integer_range<Integer>::cspan(const debug_vector<T>& range) const {
	return this->span(range);
}


template<typename Integer>
integer_range<Integer>::integer_range(const Integer begin, const Integer end) : m_begin{ begin }, m_end{ end } {
}
