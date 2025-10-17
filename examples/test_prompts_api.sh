#!/bin/bash
# Example script to test the Prompts API endpoints

BASE_URL="http://localhost:8001"

echo "=================================================="
echo "Testing Prompts API Endpoints"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: List all prompts
echo -e "${BLUE}1. Listing all latest prompts...${NC}"
curl -s -X GET "${BASE_URL}/prompts" | jq '.'
echo ""
echo ""

# Test 2: List prompts filtered by technique
echo -e "${BLUE}2. Listing Chain-of-Thought (cot) prompts...${NC}"
curl -s -X GET "${BASE_URL}/prompts?technique_key=cot" | jq '.'
echo ""
echo ""

# Test 3: Create a new prompt
echo -e "${BLUE}3. Creating a new prompt...${NC}"
CREATE_RESPONSE=$(curl -s -X POST "${BASE_URL}/prompts" \
  -H "Content-Type: application/json" \
  -d '{
    "technique_key": "zero-shot",
    "title": "API Test Prompt",
    "language": "en",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Translate this to {{language}}: {{text}}"}
    ],
    "variables": [
      {"name": "text", "type": "string", "required": true, "desc": "Text to translate"},
      {"name": "language", "type": "string", "required": true, "desc": "Target language"}
    ],
    "model_hint": "Any multilingual model"
  }')

echo "$CREATE_RESPONSE" | jq '.'
PROMPT_ID=$(echo "$CREATE_RESPONSE" | jq -r '.example_id')
echo ""
echo -e "${GREEN}Created prompt with ID: ${PROMPT_ID}${NC}"
echo ""
echo ""

# Test 4: Get the created prompt
echo -e "${BLUE}4. Getting the created prompt by ID...${NC}"
curl -s -X GET "${BASE_URL}/prompts/${PROMPT_ID}" | jq '.'
echo ""
echo ""

# Test 5: Render the prompt with variables
echo -e "${BLUE}5. Rendering the prompt with variables...${NC}"
curl -s -X POST "${BASE_URL}/prompts/${PROMPT_ID}/render" \
  -H "Content-Type: application/json" \
  -d '{
    "variables": {
      "text": "Hello, world!",
      "language": "Spanish"
    }
  }' | jq '.'
echo ""
echo ""

# Test 6: Test rendering with missing variables
echo -e "${BLUE}6. Testing render with missing variables (should show warnings)...${NC}"
curl -s -X POST "${BASE_URL}/prompts/${PROMPT_ID}/render" \
  -H "Content-Type: application/json" \
  -d '{
    "variables": {
      "text": "Hello, world!"
    }
  }' | jq '.'
echo ""
echo ""

# Test 7: Create a new version
echo -e "${BLUE}7. Creating a new version of the prompt...${NC}"
VERSION_RESPONSE=$(curl -s -X POST "${BASE_URL}/prompts/${PROMPT_ID}/versions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are an expert translator."},
      {"role": "user", "content": "Translate to {{language}}: {{text}}"}
    ],
    "model_hint": "GPT-4 or Claude recommended"
  }')

echo "$VERSION_RESPONSE" | jq '.'
NEW_VERSION_ID=$(echo "$VERSION_RESPONSE" | jq -r '.example_id')
echo ""
echo -e "${GREEN}Created new version with ID: ${NEW_VERSION_ID}${NC}"
echo ""
echo ""

# Test 8: Get latest by technique and title
echo -e "${BLUE}8. Getting latest version by technique + title...${NC}"
curl -s -X GET "${BASE_URL}/prompts/zero-shot/API%20Test%20Prompt/latest" | jq '.'
echo ""
echo ""

# Test 9: Disable the prompt
echo -e "${BLUE}9. Disabling the prompt...${NC}"
curl -s -X PATCH "${BASE_URL}/prompts/${NEW_VERSION_ID}" \
  -H "Content-Type: application/json" \
  -d '{
    "is_enabled": false
  }' | jq '.'
echo ""
echo ""

# Test 10: Archive the prompt
echo -e "${BLUE}10. Archiving the prompt...${NC}"
curl -s -X PATCH "${BASE_URL}/prompts/${NEW_VERSION_ID}" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "archived"
  }' | jq '.'
echo ""
echo ""

# Test 11: List enabled prompts (should not include our archived prompt)
echo -e "${BLUE}11. Listing only enabled prompts...${NC}"
curl -s -X GET "${BASE_URL}/prompts?enabled=true" | jq '.count'
echo ""
echo ""

# Test 12: Get an existing prompt and test rendering
echo -e "${BLUE}12. Testing with an existing CoT prompt...${NC}"
COT_PROMPT=$(curl -s -X GET "${BASE_URL}/prompts?technique_key=cot" | jq -r '.prompts[0].example_id')

if [ "$COT_PROMPT" != "null" ] && [ -n "$COT_PROMPT" ]; then
    echo -e "${GREEN}Found CoT prompt: ${COT_PROMPT}${NC}"
    
    echo -e "${YELLOW}Rendering with variables...${NC}"
    curl -s -X POST "${BASE_URL}/prompts/${COT_PROMPT}/render" \
      -H "Content-Type: application/json" \
      -d '{
        "variables": {
          "expression": "42 * 13 + 7"
        }
      }' | jq '.'
else
    echo -e "${YELLOW}No CoT prompts found, skipping...${NC}"
fi

echo ""
echo ""
echo -e "${GREEN}=================================================="
echo "All tests completed!"
echo "==================================================${NC}"

