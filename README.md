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
python main.py
```

### Set up docker env:
For setup, you can either pull the docker image and run it or you can build it on your own.
1) Pulling the docker image from my account:
```shell
docker pull ajevnisek/single-image-bg-detector:v1
docker run -v $(pwd)/docker_results:/home/code/results -it ajevnisek/single-image-bg-detector:v1 /bin/bash
```
and when the docker runs, hit:
```shell
python main.py
```
2) Building the docker image on your own:
```shell
docker build -t <your-docker.io-username>/single-image-bg-detector --file docker/Dockerfile .
docker tag <your-docker.io-username>/single-image-bg-detector <your-docker.io-username>/single-image-bg-detector:v1
docker push <your-docker.io-username>/single-image-bg-detector:v1
```

Run the docker container:
```shell
docker run -v $(pwd)/docker_results:/home/code/results -it <your-docker.io-username>/single-image-bg-detector:v1 /bin/bash
```
and when the docker runs, hit:
```shell
python main.py
```
