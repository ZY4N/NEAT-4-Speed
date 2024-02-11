#pragma once

#include <span>
#include <cassert>
#include "util/debug_vector.hpp"

template<class T, std::size_t Extent = std::dynamic_extent>
class debug_span : public std::span<T, Extent> {
public:
	using parent_type = std::span<T, Extent>;
	using parent_type::span;

	using element_type = parent_type::element_type;
	using value_type = parent_type::value_type;
	using size_type = parent_type::size_type;
	using difference_type = parent_type::difference_type;
	using pointer = parent_type::pointer;
	using const_pointer = parent_type::const_pointer;
	using reference = parent_type::reference;
	using const_reference = parent_type::const_reference;
	using iterator = parent_type::iterator;
	using const_iterator = parent_type::const_iterator;
	using reverse_iterator = parent_type::reverse_iterator;
	using const_reverse_iterator = parent_type::const_reverse_iterator;

	static constexpr std::size_t extent = Extent;

	inline constexpr reference operator[](const size_type idx ) const {
		assert(idx < this->size());
		return (*static_cast<const parent_type*>(this))[idx];
	}
};

template<class It, class EndOrSize>
debug_span(It, EndOrSize) -> debug_span<std::remove_reference_t<std::iter_reference_t<It>>>;

template<class T, std::size_t N>
debug_span(T (&)[N]) -> debug_span<T, N>;

template<class T, std::size_t N>
debug_span(std::array<T, N>&) -> debug_span<T, N>;

template<class T, std::size_t N>
debug_span(const std::array<T, N>&) -> debug_span<const T, N>;

template<class T>
debug_span(const debug_vector<T>&) -> debug_span<const T>;

template<class T>
debug_span(debug_vector<T>&) -> debug_span<T>;


template<class R>
debug_span(R&&) -> debug_span<std::remove_reference_t<std::ranges::range_reference_t<R>>>;
