#!/usr/bin/env bash

echo "world size: ${WORLD_SIZE:=${SLURM_NTASKS:-}}"
echo "global rank: ${RANK:=${SLURM_PROCID:-}}"
echo "local rank: ${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
export WORLD_SIZE
export RANK
export LOCAL_RANK

#####
# Adapted from https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
# If this is randomly not correct, just try again/another value
export MASTER_PORT=12354
echo "MASTER_PORT=$MASTER_PORT"

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST=${SLURM_NODELIST}"

# if [ ${SLURM_NODELIST:4:1} == "," ]; then
#     echo "MASTER_ADDR=${SLURM_NODELIST:0:4}"
#     export MASTER_ADDR=${SLURM_NODELIST:0:4}
# elif [ ${SLURM_NODELIST:3:1} == "[" ]; then
#     echo "MASTER_ADDR=${SLURM_NODELIST:0:3}${SLURM_NODELIST:4:4}"
#     export MASTER_ADDR=${SLURM_NODELIST:0:3}${SLURM_NODELIST:4:4}
# else
#     echo "MASTER_ADDR=${SLURM_NODELIST}"
#     export MASTER_ADDR=${SLURM_NODELIST}
# fi
#
#####

if [[ ${SLURM_NODELIST} == *"["*"]" ]]; then
    INDICES=$(echo ${SLURM_NODELIST} | cut -d \[ -f 2)
    if [[ ${INDICES} == *","* ]]; then
        INDICES=$(echo ${INDICES} | cut -d , -f 1)
        if [[ ${INDICES} == *"-"* ]]; then
            INDICES=$(echo ${INDICES} | cut -d - -f 1)
        fi
    elif [[ ${INDICES} == *"-"* ]]; then
        INDICES=$(echo ${INDICES} | cut -d - -f 1)
    else
        INDICES=$(echo ${INDICES} | cut -d \] -f 1)
    fi

    export MASTER_ADDR=$(echo ${SLURM_NODELIST} | cut -d \[ -f 1)$INDICES
elif [[ ${SLURM_NODELIST} == *","* ]]; then
    export MASTER_ADDR=$(echo ${SLURM_NODELIST} | cut -d , -f 1)
else
    export MASTER_ADDR=${SLURM_NODELIST}
fi

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR
