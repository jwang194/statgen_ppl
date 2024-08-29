python3 generate_am.py ${1} ${2} ${3} ${4} ${5}
echo "Data generated."
python3 distribute_benchmark.py 'am' ${1} ${2} ${3} ${4} ${5}
echo "Inference completed."
python3 extractor.py 'am' ${1} ${2} ${3} ${4} ${5}
echo "Done."
