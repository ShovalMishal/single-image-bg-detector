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
Setup
```shell
docker build -t ajevnisek/single-image-bg-detector --file docker/Dockerfile .
docker tag ajevnisek/single-image-bg-detector ajevnisek/single-image-bg-detector:v1
docker push ajevnisek/single-image-bg-detector:v1
```

Run:
```shell
docker run -v $(pwd)/docker_results:/home/code/results -it ajevnisek/single-image-bg-detector:v1 /bin/bash
```
and when the docker runs, hit:
```shell
python main.py
```
