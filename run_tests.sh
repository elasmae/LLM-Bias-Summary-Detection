#!/bin/bash

echo "🔍 Running tests with pytest..."

pytest tests/ --tb=short --disable-warnings

echo "✅ All tests finished."
