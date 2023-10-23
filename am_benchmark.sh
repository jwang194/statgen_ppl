python generate_am.py ${1} ${2} ${3} ${4}
echo "Data generated."
python distribute_benchmark.py 'am' ${1} ${2} ${3} ${4} ${5}
echo "Inference completed."
python extractor.py 'am' ${1} ${2} ${3} ${4} ${5}
echo "Done."
