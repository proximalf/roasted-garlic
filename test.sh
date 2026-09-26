echo "-- TEST FORMAT --"
uv run tests/test_format.py

echo "-- TEST HISTOGRAM --"
uv run tests/test_histogram.py

echo "-- TEST OPERATIONS --"
uv run tests/test_operations.py

echo "-- TEST SELECTION --"
uv run tests/test_selection.py