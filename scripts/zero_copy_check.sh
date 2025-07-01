#!/bin/bash

# Zero-Copy Verification Script for Psyne
# This script analyzes the C++ codebase for potential zero-copy violations

echo "🔍 Psyne Zero-Copy Analysis"
echo "=========================="
echo ""

# Color codes for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
POTENTIAL_ISSUES=0
WARNINGS=0
GOOD_PRACTICES=0

# Function to check a pattern and report results
check_pattern() {
    local pattern="$1"
    local description="$2"
    local severity="$3"  # ERROR, WARNING, or INFO
    local files="$4"
    
    if [ -z "$files" ]; then
        files="include/psyne/*.hpp src/**/*.cpp src/**/*.hpp"
    fi
    
    echo -e "${BLUE}Checking: $description${NC}"
    
    # Use find to get all files, then grep them
    local results=$(find . -path "./include/psyne/*.hpp" -o -path "./src/*.cpp" -o -path "./src/*.hpp" -o -path "./src/*/*.cpp" -o -path "./src/*/*.hpp" | xargs grep -n "$pattern" 2>/dev/null || true)
    
    if [ ! -z "$results" ]; then
        if [ "$severity" = "ERROR" ]; then
            echo -e "${RED}❌ POTENTIAL ZERO-COPY VIOLATION:${NC}"
            ((POTENTIAL_ISSUES++))
        elif [ "$severity" = "WARNING" ]; then
            echo -e "${YELLOW}⚠️  WARNING:${NC}"
            ((WARNINGS++))
        else
            echo -e "${GREEN}ℹ️  INFO:${NC}"
        fi
        
        echo "$results" | while read -r line; do
            echo "   $line"
        done
        echo ""
    else
        if [ "$severity" = "ERROR" ]; then
            echo -e "${GREEN}✅ No violations found${NC}"
        elif [ "$severity" = "WARNING" ]; then
            echo -e "${GREEN}✅ No issues found${NC}"
        else
            echo -e "${GREEN}✅ Check passed${NC}"
        fi
        ((GOOD_PRACTICES++))
        echo ""
    fi
}

echo "🚀 Starting Zero-Copy Analysis..."
echo ""

# 1. Check for explicit copying operations that could violate zero-copy
echo "=== CRITICAL ZERO-COPY CHECKS ==="
echo ""

check_pattern "std::copy\(" "Explicit std::copy calls (potential data copying)" "ERROR"
check_pattern "memcpy\(" "Direct memcpy calls (potential data copying)" "ERROR"
check_pattern "\.copy\(" "Object copy method calls" "WARNING"
check_pattern "= .*\.data\(\)" "Direct data copying assignments" "WARNING"

# 2. Check for unnecessary string/vector copying
echo "=== STRING AND CONTAINER COPYING ==="
echo ""

check_pattern "std::string.*=" "String assignments (check if copying)" "WARNING"
check_pattern "std::vector.*=" "Vector assignments (check if copying)" "WARNING"
check_pattern "return.*\.substr\(" "Substring operations (creates copies)" "WARNING"
check_pattern "\.push_back\(.*std::string" "String push_back operations" "INFO"

# 3. Check for proper move semantics usage
echo "=== MOVE SEMANTICS VERIFICATION ==="
echo ""

check_pattern "std::move\(" "Move semantics usage (review necessity)" "WARNING"
check_pattern "operator=" "Assignment operator overloads (potential copying)" "WARNING"
check_pattern "&&" "Rvalue references (enables zero-copy)" "INFO"
check_pattern "= std::make_unique" "Unique pointer usage (good practice)" "INFO"

# 4. Check for buffer management patterns
echo "=== BUFFER MANAGEMENT ==="
echo ""

check_pattern "new \[\]" "Raw array allocations (check if necessary)" "WARNING"
check_pattern "malloc\(" "Direct malloc usage (check if necessary)" "WARNING"
check_pattern "realloc\(" "Memory reallocation (potential copying)" "ERROR"
check_pattern "\.resize\(" "Container resizing (potential copying)" "WARNING"

# 5. Check for proper zero-copy message patterns
echo "=== MESSAGE ZERO-COPY PATTERNS ==="
echo ""

check_pattern "data_\(" "Data pointer access (good for zero-copy)" "INFO"
check_pattern "reserve_space\(" "Space reservation (zero-copy pattern)" "INFO"
check_pattern "commit_message\(" "Message commitment (zero-copy pattern)" "INFO"
check_pattern "\.begin\(\)" "Iterator usage (potentially zero-copy)" "INFO"

# 6. Check for problematic return patterns
echo "=== RETURN VALUE ANALYSIS ==="
echo ""

check_pattern "return std::string\(" "String returns (check if move-optimized)" "WARNING"
check_pattern "return std::vector\(" "Vector returns (check if move-optimized)" "WARNING"
check_pattern "return.*\.data\(\)" "Data pointer returns (good for zero-copy)" "INFO"

# 7. Check for const-correctness (helps prevent copying)
echo "=== CONST-CORRECTNESS ==="
echo ""

check_pattern "const.*&" "Const references (good for zero-copy)" "INFO"
check_pattern "const.*\*" "Const pointers (good for zero-copy)" "INFO"

# 8. Look for specific zero-copy optimizations
echo "=== ZERO-COPY OPTIMIZATIONS ==="
echo ""

check_pattern "reinterpret_cast" "Reinterpret casts (zero-copy technique)" "INFO"
check_pattern "static_cast.*uint8_t\*" "Byte pointer casts (zero-copy access)" "INFO"
check_pattern "\.data\(\) \+" "Pointer arithmetic on data (zero-copy)" "INFO"

# 9. Check for problematic temporary objects
echo "=== TEMPORARY OBJECT ANALYSIS ==="
echo ""

check_pattern "auto.*=.*std::string\(" "Temporary string creation" "WARNING"
check_pattern "auto.*=.*std::vector\(" "Temporary vector creation" "WARNING"

# 10. Message template analysis
echo "=== MESSAGE TEMPLATE ANALYSIS ==="
echo ""

check_pattern "template.*Message" "Message templates (should be zero-copy)" "INFO"
check_pattern "Message<.*>::Message" "Message constructors" "INFO"

echo ""
echo "=== ANALYSIS SUMMARY ==="
echo ""

if [ $POTENTIAL_ISSUES -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 EXCELLENT! No zero-copy violations detected!${NC}"
    echo -e "${GREEN}✅ Psyne appears to maintain zero-copy semantics throughout${NC}"
elif [ $POTENTIAL_ISSUES -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Some warnings found, but no critical violations${NC}"
    echo -e "${YELLOW}📋 Review warnings to ensure optimal zero-copy performance${NC}"
else
    echo -e "${RED}❌ ATTENTION: $POTENTIAL_ISSUES potential zero-copy violations found${NC}"
    echo -e "${RED}🔧 These should be reviewed and addressed${NC}"
fi

echo ""
echo "📊 Statistics:"
echo "   Potential Issues: $POTENTIAL_ISSUES"
echo "   Warnings: $WARNINGS"
echo "   Good Practices Found: $GOOD_PRACTICES"
echo ""

# Additional specific checks for Psyne patterns
echo "=== PSYNE-SPECIFIC CHECKS ==="
echo ""

echo -e "${BLUE}Checking Ring Buffer Implementation...${NC}"
RING_BUFFER_ISSUES=$(find . -name "ring_buffer*.cpp" -o -name "ring_buffer*.hpp" | xargs grep -n "memcpy\|std::copy" 2>/dev/null || true)
if [ -z "$RING_BUFFER_ISSUES" ]; then
    echo -e "${GREEN}✅ Ring buffers appear to be zero-copy${NC}"
else
    echo -e "${YELLOW}⚠️  Ring buffer implementation may have copying:${NC}"
    echo "$RING_BUFFER_ISSUES"
    ((WARNINGS++))
fi
echo ""

echo -e "${BLUE}Checking Message Implementation...${NC}"
MESSAGE_ISSUES=$(find . -name "message*.cpp" -o -name "message*.hpp" | xargs grep -n "memcpy\|std::copy" 2>/dev/null || true)
if [ -z "$MESSAGE_ISSUES" ]; then
    echo -e "${GREEN}✅ Message implementation appears to be zero-copy${NC}"
else
    echo -e "${YELLOW}⚠️  Message implementation may have copying:${NC}"
    echo "$MESSAGE_ISSUES"
    ((WARNINGS++))
fi
echo ""

echo -e "${BLUE}Checking Channel Implementation...${NC}"
CHANNEL_ISSUES=$(find . -name "*channel*.cpp" -o -name "*channel*.hpp" | xargs grep -n "memcpy\|std::copy" 2>/dev/null || true)
if [ -z "$CHANNEL_ISSUES" ]; then
    echo -e "${GREEN}✅ Channel implementation appears to be zero-copy${NC}"
else
    echo -e "${YELLOW}⚠️  Channel implementation may have copying:${NC}"
    echo "$CHANNEL_ISSUES"
    ((WARNINGS++))
fi
echo ""

# Final recommendations
echo "=== RECOMMENDATIONS ==="
echo ""
echo "🎯 Zero-Copy Best Practices for Psyne:"
echo "   • Use const references instead of pass-by-value"
echo "   • Prefer std::move() for temporary objects"
echo "   • Use reinterpret_cast for type punning instead of copying"
echo "   • Return pointers or references to existing data"
echo "   • Avoid std::copy, memcpy except when absolutely necessary"
echo "   • Use placement new instead of copying for initialization"
echo ""

if [ $POTENTIAL_ISSUES -eq 0 ]; then
    echo -e "${GREEN}🚀 Psyne maintains excellent zero-copy principles!${NC}"
    exit 0
else
    echo -e "${RED}🔧 Please review potential violations before release${NC}"
    exit 1
fi