python3 generate_lmm.py ${2} ${3} ${4} ${5}
echo "Data generated."
python3 distribute_benchmark.py 'lmm' ${1} ${2} ${3} ${4} ${5}
echo "Inference completed."
python3 extractor.py 'lmm' ${1} ${2} ${3} ${4} ${5}
echo "Done."
