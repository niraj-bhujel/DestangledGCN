#!/bin/bash
#PBS -l select=1:ncpus=10:ngpus=1:host=dgx03
#PBS -l software=nvidia-docker
#PBS -N dgx03
#PBS -l walltime=72:00:00

cd /home/i2r/stunb/scratch/Recon_GCN

IMAGE="stunb/pytorch:cuda_10.1_py3.6"
docker build -t $IMAGE .

# Initialize default arguments
NAME="stunb_pytorch" # container name

#Mount dir to save training output, otherwise saved result will deleted once container is exited.
#!Mount path MUST be same as in output dir mentioned in Dockerfile
VOLUME="$HOME/Desktop/Recon_GCN/out/:/Recon_GCN/out/" 

# arguments like --rm
EXTRA_DOCKER_ARGS="$@" 

# command to run script
COMMAND="python3.6 train.py --trial 1 --run 2 --mode search --prefix dgx"
# Run the Docker container in interactive mode
echo "starting container..."
nvidia-docker run --rm \
				  --name=$NAME \
				  --volume=$VOLUME \
				  --user "$(id -u):$(id -g)" \
				  $EXTRA_DOCKER_ARGS \
				  $IMAGE \
				  $COMMAND