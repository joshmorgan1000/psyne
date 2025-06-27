#!/bin/bash
# Script to check for zero-copy violations in Psyne codebase

echo "Checking for zero-copy violations..."
echo "===================================="

# Check for memcpy usage (excluding allowed cases)
echo -e "\n1. Checking for memcpy usage:"
rg -n "memcpy|std::copy|std::copy_n" \
   --type cpp \
   --glob '!*test*' \
   --glob '!*benchmark*' \
   --glob '!*example*' \
   include/ src/ | \
   grep -v "DEPRECATED" | \
   grep -v "deprecated" || echo "✓ No unauthorized memcpy found"

# Check for copy constructors in message types
echo -e "\n2. Checking for copy constructors in messages:"
rg -n "Message.*\(.*const.*Message.*&" \
   --type cpp \
   include/psyne/core/ || echo "✓ No copy constructors found"

# Check for std::vector in hot paths
echo -e "\n3. Checking for std::vector in core components:"
rg -n "std::vector" \
   --type cpp \
   include/psyne/core/ \
   include/psyne/memory/ | \
   grep -v "deprecated" || echo "✓ No std::vector in hot paths"

# Check for dynamic allocations
echo -e "\n4. Checking for dynamic allocations:"
rg -n "new\s|malloc|calloc" \
   --type cpp \
   include/psyne/core/ \
   include/psyne/memory/ | \
   grep -v "placement new" | \
   grep -v "aligned_alloc" || echo "✓ No unexpected dynamic allocations"

# Check for proper alignment
echo -e "\n5. Checking for alignment directives:"
rg -n "alignas|aligned" \
   --type cpp \
   include/psyne/memory/ | \
   wc -l | \
   awk '{if ($1 > 0) print "✓ Found " $1 " alignment directives"; else print "⚠ No alignment directives found"}'

# Check for atomic usage in SPSC
echo -e "\n6. Checking SPSC implementation:"
rg -A5 -B5 "class RingBuffer<struct SingleProducer, struct SingleConsumer>" \
   include/psyne/memory/ring_buffer.hpp | \
   rg "atomic" | \
   wc -l | \
   awk '{if ($1 == 0) print "✓ SPSC uses no atomics"; else print "⚠ SPSC still uses " $1 " atomic operations"}'

echo -e "\n===================================="
echo "Zero-copy compliance check complete!"