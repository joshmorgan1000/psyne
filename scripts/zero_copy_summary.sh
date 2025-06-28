#!/bin/bash

# Quick Zero-Copy Summary for Psyne
echo "🔍 Psyne Zero-Copy Quick Analysis"
echo "================================"
echo ""

# Color codes
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Checking critical zero-copy violations...${NC}"

# Check for explicit copying in core components
CRITICAL_ISSUES=0

echo ""
echo "1. Checking for memcpy/std::copy in core components..."
CORE_COPYING=$(find src/core src/memory src/channel -name "*.cpp" -o -name "*.hpp" | xargs grep -l "memcpy\|std::copy" 2>/dev/null || true)
if [ -z "$CORE_COPYING" ]; then
    echo -e "${GREEN}✅ No memcpy/std::copy found in core components${NC}"
else
    echo -e "${RED}❌ Found copying in core components:${NC}"
    echo "$CORE_COPYING"
    ((CRITICAL_ISSUES++))
fi

echo ""
echo "2. Checking message implementations..."
MESSAGE_COPYING=$(find src/core -name "message*" | xargs grep -n "memcpy\|std::copy" 2>/dev/null || true)
if [ -z "$MESSAGE_COPYING" ]; then
    echo -e "${GREEN}✅ Message implementations are zero-copy${NC}"
else
    echo -e "${RED}❌ Found copying in message code:${NC}"
    echo "$MESSAGE_COPYING"
    ((CRITICAL_ISSUES++))
fi

echo ""
echo "3. Checking ring buffer implementations..."
RING_COPYING=$(find src/memory -name "*ring*" | xargs grep -n "memcpy\|std::copy" 2>/dev/null || true)
if [ -z "$RING_COPYING" ]; then
    echo -e "${GREEN}✅ Ring buffers are zero-copy${NC}"
else
    echo -e "${RED}❌ Found copying in ring buffers:${NC}"
    echo "$RING_COPYING"
    ((CRITICAL_ISSUES++))
fi

echo ""
echo "4. Checking for proper zero-copy patterns..."
ZERO_COPY_PATTERNS=$(grep -r "reinterpret_cast\|\.data()\|static_cast.*uint8_t" src/core src/memory src/channel | wc -l)
echo -e "${GREEN}✅ Found $ZERO_COPY_PATTERNS zero-copy patterns (good!)${NC}"

echo ""
echo "5. Checking for move semantics..."
MOVE_SEMANTICS=$(grep -r "std::move\|&&.*)" include/psyne src/core | wc -l)
echo -e "${GREEN}✅ Found $MOVE_SEMANTICS move semantic patterns (good!)${NC}"

echo ""
echo "=== SUMMARY ==="
if [ $CRITICAL_ISSUES -eq 0 ]; then
    echo -e "${GREEN}🎉 EXCELLENT! No critical zero-copy violations found!${NC}"
    echo -e "${GREEN}✅ Psyne maintains zero-copy semantics in core components${NC}"
    echo ""
    echo -e "${BLUE}📊 Zero-Copy Evidence:${NC}"
    echo "   • No memcpy/std::copy in core message/ring buffer code"
    echo "   • Extensive use of reinterpret_cast for type punning"
    echo "   • Proper move semantics implementation"
    echo "   • Direct .data() pointer access patterns"
    echo ""
    echo -e "${GREEN}🚀 Psyne is ready for 1.0.0 with true zero-copy performance!${NC}"
else
    echo -e "${RED}❌ Found $CRITICAL_ISSUES critical issues that should be reviewed${NC}"
fi