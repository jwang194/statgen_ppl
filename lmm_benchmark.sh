python generate_lmm.py ${1} ${2} ${3} ${4}
echo "Data generated."
python distribute_benchmark.py 'lmm' ${1} ${2} ${3} ${4}
echo "Inference completed."
python extractor.py 'lmm' ${1} ${2} ${3} ${4}
echo "Done."
