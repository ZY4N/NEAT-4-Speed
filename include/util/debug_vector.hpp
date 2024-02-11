#pragma once

#include <ranges>
#include <vector>
#include <cassert>

template<class T, class Allocator = std::allocator<T>>
class debug_vector : public std::vector<T, Allocator> {
public:
	using parent_type = std::vector<T, Allocator>;
	using value_type = parent_type::value_type;
	using allocator_type = parent_type::allocator_type;
	using size_type = parent_type::size_type;
	using difference_type = parent_type::difference_type;
	using reference = parent_type::reference;
	using const_reference = parent_type::const_reference;
	using pointer = parent_type::pointer;
	using const_pointer = parent_type::const_pointer;
	using iterator = parent_type::iterator;
	using const_iterator = parent_type::const_iterator;
	using reverse_iterator = parent_type::reverse_iterator;
	using const_reverse_iterator = parent_type::const_reverse_iterator;

	using parent_type::vector;

	inline constexpr reference operator[](const size_type idx) {
		assert(idx < this->size());
		return (*static_cast<parent_type*>(this))[idx];
	}

	inline constexpr const_reference operator[](const size_type idx) const {
		assert(idx < this->size());
		return (*static_cast<const parent_type*>(this))[idx];
	}
};

template<class InputIt, class Alloc = std::allocator<typename std::iterator_traits<InputIt>::value_type>>
debug_vector(InputIt, InputIt, Alloc = Alloc())
	-> debug_vector<typename std::iterator_traits<InputIt>::value_type, Alloc>;
/*
template<std::ranges::input_range R, class Alloc = std::allocator<std::ranges::range_value_t<R>>>
debug_vector(std::from_range_t, R&&, Alloc = Alloc()) -> debug_vector<std::ranges::range_value_t<R>, Alloc>;
*/
