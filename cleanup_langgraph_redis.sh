#!/bin/bash

# Safe cleanup script for LangGraph-related Redis data
# This preserves all other application data in the shared Redis instance

echo "🔍 Analyzing LangGraph Redis data..."

# Count what we're about to delete
CHECKPOINT_COUNT=$(docker exec redis-stack redis-cli KEYS "checkpoint*" | wc -l)
STORE_COUNT=$(docker exec redis-stack redis-cli KEYS "store:*" | wc -l)

echo "📊 Found:"
echo "  - $CHECKPOINT_COUNT checkpoint keys (LangGraph conversation state)"
echo "  - $STORE_COUNT store keys (Langmem memories)"

read -p "❓ Do you want to proceed with deletion? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

echo ""
echo "🗑️  Deleting LangGraph checkpoints..."
docker exec redis-stack redis-cli --scan --pattern "checkpoint*" | xargs -L 100 docker exec redis-stack redis-cli DEL
docker exec redis-stack redis-cli --scan --pattern "checkpoint_*" | xargs -L 100 docker exec redis-stack redis-cli DEL

echo "🗑️  Deleting Langmem stores..."
docker exec redis-stack redis-cli --scan --pattern "store:*" | xargs -L 100 docker exec redis-stack redis-cli DEL

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Verifying cleanup..."
REMAINING_CHECKPOINTS=$(docker exec redis-stack redis-cli KEYS "checkpoint*" | wc -l)
REMAINING_STORES=$(docker exec redis-stack redis-cli KEYS "store:*" | wc -l)

echo "  - Remaining checkpoint keys: $REMAINING_CHECKPOINTS"
echo "  - Remaining store keys: $REMAINING_STORES"

if [ "$REMAINING_CHECKPOINTS" -eq 0 ] && [ "$REMAINING_STORES" -eq 0 ]; then
    echo ""
    echo "✅ All LangGraph data successfully removed!"
    echo "🔄 You can now restart your LangGraph server"
else
    echo ""
    echo "⚠️  Warning: Some keys still remain. You may need to run cleanup again."
fi
