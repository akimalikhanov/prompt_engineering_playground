#!/bin/bash
# Script to run the API server

set -e

echo "Starting API server..."
echo "API will be available at: http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Please create a .env file with database configuration"
    exit 1
fi

# Load environment variables
source .env

# Run the API server (only watch specific directories)
uvicorn api.main:app --reload --port 8000 --host 0.0.0.0 \
  --reload-dir api \
  --reload-dir schemas \
  --reload-dir models \
  --reload-dir services \
  --reload-dir utils \
  --reload-dir config

