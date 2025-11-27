#!/usr/bin/env bash
# Script to verify Loki is working correctly

set -euo pipefail

LOKI_PORT="${LOKI_HTTP_PORT:-3100}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"

echo "🔍 Verifying Loki setup..."
echo ""

# 1. Check if Loki container is running
echo "1️⃣ Checking Loki container status..."
if docker ps --format "{{.Names}}" | grep -q "^loki$"; then
    echo "   ✅ Loki container is running"
    LOKI_STATUS=$(docker inspect -f '{{.State.Status}}' loki 2>/dev/null || echo "not found")
    echo "   Status: $LOKI_STATUS"
else
    echo "   ❌ Loki container is NOT running"
    echo "   Start it with: docker compose -f infra/docker-compose.yml up -d loki"
    exit 1
fi
echo ""

# 2. Check Loki logs for errors
echo "2️⃣ Checking Loki logs for recent errors..."
RECENT_ERRORS=$(docker logs loki --tail 20 2>&1 | grep -i "error" | tail -5 || echo "")
if [ -z "$RECENT_ERRORS" ]; then
    echo "   ✅ No recent errors found"
else
    echo "   ⚠️  Recent errors found:"
    echo "$RECENT_ERRORS" | sed 's/^/      /'
fi
echo ""

# 3. Test Loki HTTP API - Ready endpoint
echo "3️⃣ Testing Loki HTTP API (ready endpoint)..."
if curl -fsS "http://localhost:${LOKI_PORT}/ready" >/dev/null 2>&1; then
    echo "   ✅ Loki ready endpoint responds"
else
    echo "   ❌ Loki ready endpoint not responding"
    echo "   Try: curl http://localhost:${LOKI_PORT}/ready"
fi
echo ""

# 4. Test Loki HTTP API - Metrics endpoint
echo "4️⃣ Testing Loki metrics endpoint..."
if curl -fsS "http://localhost:${LOKI_PORT}/metrics" >/dev/null 2>&1; then
    echo "   ✅ Loki metrics endpoint responds"
    METRIC_COUNT=$(curl -fsS "http://localhost:${LOKI_PORT}/metrics" 2>/dev/null | wc -l)
    echo "   Found $METRIC_COUNT metric lines"
else
    echo "   ❌ Loki metrics endpoint not responding"
fi
echo ""

# 5. Test Loki query API
echo "5️⃣ Testing Loki query API..."
QUERY_RESPONSE=$(curl -fsS "http://localhost:${LOKI_PORT}/loki/api/v1/labels" 2>&1 || echo "ERROR")
if echo "$QUERY_RESPONSE" | grep -q "ERROR\|error"; then
    echo "   ⚠️  Query API test failed"
    echo "   Response: $QUERY_RESPONSE"
else
    echo "   ✅ Loki query API responds"
    LABEL_COUNT=$(echo "$QUERY_RESPONSE" | grep -o '"[^"]*"' | wc -l || echo "0")
    echo "   Found $LABEL_COUNT labels"
fi
echo ""

# 6. Check MinIO bucket exists
echo "6️⃣ Checking MinIO bucket for Loki..."
if docker ps --format "{{.Names}}" | grep -q "^mlflow-minio$"; then
    BUCKET_NAME="${LOKI_BUCKET:-loki-logs}"
    # Check if bucket directory exists in MinIO container
    if docker exec mlflow-minio test -d "/data/${BUCKET_NAME}" 2>/dev/null; then
        echo "   ✅ MinIO bucket '${BUCKET_NAME}' exists"
        # Show bucket contents count
        FILE_COUNT=$(docker exec mlflow-minio find "/data/${BUCKET_NAME}" -type f 2>/dev/null | wc -l || echo "0")
        echo "   Found $FILE_COUNT files in bucket"
    else
        echo "   ⚠️  MinIO bucket '${BUCKET_NAME}' not found"
        echo "   Run: docker compose -f infra/docker-compose.yml run --rm minio-init"
    fi
else
    echo "   ⚠️  MinIO container not running, skipping bucket check"
fi
echo ""

# 7. Check Grafana datasource
echo "7️⃣ Checking Grafana datasource configuration..."
if [ -f "infra/monitoring/grafana/provisioning/datasources/grafana-datasources.yaml" ]; then
    if grep -q "type: loki" "infra/monitoring/grafana/provisioning/datasources/grafana-datasources.yaml"; then
        echo "   ✅ Loki datasource configured in Grafana"
        if docker ps --format "{{.Names}}" | grep -q "^grafana$"; then
            echo "   ✅ Grafana container is running"
            echo "   Access Grafana at: http://localhost:${GRAFANA_PORT} (admin/admin)"
            echo "   Then go to: Configuration → Data Sources → Loki"
        else
            echo "   ⚠️  Grafana container not running"
        fi
    else
        echo "   ❌ Loki datasource not found in Grafana config"
    fi
else
    echo "   ⚠️  Grafana datasource config file not found"
fi
echo ""

# 8. Send a test log entry
echo "8️⃣ Sending test log entry to Loki..."
TIMESTAMP=$(date +%s)000000000
TEST_LOG='{"streams":[{"stream":{"job":"test","level":"info"},"values":[["'${TIMESTAMP}'","Test log entry from verify script"]]}]}'
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    "http://localhost:${LOKI_PORT}/loki/api/v1/push" \
    -d "$TEST_LOG" 2>&1 || echo "ERROR\n000")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "204" ] || [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Test log entry sent successfully (HTTP $HTTP_CODE)"
    echo "   Query it in Grafana with: {job=\"test\"}"
else
    echo "   ❌ Failed to send test log (HTTP $HTTP_CODE)"
    echo "   Response: $(echo "$RESPONSE" | head -1)"
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To verify in Grafana:"
echo "  1. Open http://localhost:${GRAFANA_PORT}"
echo "  2. Login: admin / admin"
echo "  3. Go to: Explore → Select 'Loki' datasource"
echo "  4. Try queries like:"
echo "     - {job=\"test\"}  (to see test log)"
echo "     - {container_name=\"loki\"}  (Loki's own logs)"
echo "     - {job=~\".+\"}  (all logs)"
echo ""
echo "To check Loki API directly:"
echo "  curl http://localhost:${LOKI_PORT}/ready"
echo "  curl http://localhost:${LOKI_PORT}/loki/api/v1/labels"
echo "  curl http://localhost:${LOKI_PORT}/metrics"
echo ""

