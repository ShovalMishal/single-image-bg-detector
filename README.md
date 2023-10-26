# Single Image Background Detector
*Important Note:*: clone this repo using:
```shell
git clone --recursive https://github.com/ajevnisek/single-image-bg-detector.git
```
## Install:
create conda environment:
```shell
conda env create -f environment.yml
conda activate single-image-bg-detector
```
## Usage:
```shell
SOURCE_DIR=/path/to/source/directory
TARGET_DIR=/path/to/target/directory
python main.py --source $SOURCE_DIR --target $TARGET_DIR
```

### Set up docker env:
For setup, you can either pull the docker image and run it or you can build it on your own.
1) Pulling the docker image from my account:
```shell
docker pull ajevnisek/single-image-bg-detector:v1
docker run -v $(pwd)/docker_results:/home/code/one_image_results -e SOURCE_DIR=one_image -e TARGET_DIR=one_image_results --name one-image-local-run --rm -it ajevnisek/single-image-bg-detector:v1
```

2) Building the docker image on your own:
```shell
docker build -t <your-docker.io-username>/single-image-bg-detector:base --file docker/base/Dockerfile .
docker tag <your-docker.io-username>/single-image-bg-detector:base <your-docker.io-username>/single-image-bg-detector:base
docker push <your-docker.io-username>/single-image-bg-detector:base

docker build -t <your-docker.io-username>/single-image-bg-detector:v1 --file docker/v1/Dockerfile .
docker tag <your-docker.io-username>/single-image-bg-detector:v1 <your-docker.io-username>/single-image-bg-detector:v1
docker push <your-docker.io-username>/single-image-bg-detector:v1
```

Run the docker container:
```shell
docker run -v $(pwd)/docker_results:/home/code/one_image_results -e SOURCE_DIR=one_image -e TARGET_DIR=one_image_results --name one-image-local-run --rm -it <your-docker.io-username>/single-image-bg-detector:v1
```

## Run it with runai:
```shell
local2storage $(pwd)/one_image jevnisek/single-image-bg-detector/input
runai submit --name <your_name>-single-image-bg-detector-no-gpu -g 0 -i <your-docker.io-username>/single-image-bg-detector:v1 -e SOURCE_DIR=/storage/jevnisek/single-image-bg-detector/input/one_image -e TARGET_DIR=/storage/jevnisek/single-image-bg-detector/output  --pvc=storage:/storage
```