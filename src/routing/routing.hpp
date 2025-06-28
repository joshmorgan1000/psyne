#pragma once

/**
 * @file routing.hpp
 * @brief Message routing and filtering for Psyne
 * 
 * This header provides advanced message routing and filtering capabilities
 * for Psyne channels, including type-based routing, custom filters, and
 * composite filter logic.
 * 
 * @example routing_demo.cpp
 */

#include "../../include/psyne/psyne.hpp"
#include "message_router.hpp"

namespace psyne {

/**
 * @namespace psyne::routing
 * @brief Message routing and filtering functionality
 * 
 * Provides classes and utilities for routing messages based on type,
 * size, or custom predicates. Supports creating filtered channels and
 * building complex routing logic.
 */

// Re-export main routing classes for convenience
using routing::MessageFilter;
using routing::TypeFilter;
using routing::RangeFilter;
using routing::SizeFilter;
using routing::PredicateFilter;
using routing::CompositeFilter;
using routing::MessageRouter;
using routing::FilteredChannel;
using routing::create_filtered_channel;

} // namespace psyne