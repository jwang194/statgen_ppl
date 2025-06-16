#!/bin/bash
#$ -o /u/home/j/jwang194/zp/zaitlen/statgen_ppl/logs/hpc_script.o 
#$ -e /u/home/j/jwang194/zp/zaitlen/statgen_ppl/logs/hpc_script.e
#$ -l h_rt=12:00:00
#$ -l h_data=32G
#$ -l gpu
#$ -l RTX2080Ti
#$ -l cuda=2

. /u/local/Modules/default/init/modules.sh
. /u/home/j/jwang194/.profile

module load cuda/12.3 
condaload ppl

cd /u/home/j/jwang194/zp/zaitlen/statgen_ppl

MODEL=${1}
NGPU=${2}
NLOW=${3}
NHIGH=${4}
MLOW=${5}
MHIGH=${6}
SCALE=${7}
SMART=${8}

echo 'Model: '${MODEL}', #GPUs: '${NGPU}', N bounds: '${NLOW}' '${NHIGH}', M bounds:'${MLOW}' '${MHIGH}', Scaling: '${SCALE}', Smart initialization: '${SMART}

./${MODEL}_benchmark.sh ${NGPU} ${NLOW} ${NHIGH} ${MLOW} ${MHIGH} ${SCALE} ${SMART}
