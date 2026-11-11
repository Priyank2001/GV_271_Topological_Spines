#!/bin/bash
# Activate the conda environment for this project

conda activate gv271
echo "✓ Environment 'gv271' activated"
echo "Python: $(python --version)"
echo ""
echo "Available tools:"
echo "  - python scalar_field_generator.py       Run generator examples"
echo "  - python -c 'from scalar_field_generator import ScalarFieldGenerator'  Import in Python"
