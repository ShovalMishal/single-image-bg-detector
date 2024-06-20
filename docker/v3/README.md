# Docker instructions 
# ver 3 aims to compare different vit representations
1. Build the docker:
```shell
docker build -t ad-stage1:v3 -f docker/v3/Dockerfile .
docker tag ad-stage1:v3 shovalmishal/ad-stage1:v3
docker push shovalmishal/ad-stage1:v3
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/workspace/ -e EXPORT_SCRIPT=/workspace/scripts/what_to_say.sh  -e RUN_SCRIPT=/workspace/scripts/say_hello.sh  -it shovalmishal/ad-stage1:v1


```
On runai:
```shell
runai submit --name dino-mc-value-layer0 -g 1.0 -i shovalmishal/ad-stage1:v3 -e VIT_MODEL_MODE="query" -e LAYER_NUM=0 -e MODEL_TYPE="dino_mc_vit" --pvc=storage:/storage --large-shm
runai submit --name key-layer11 -g 1.0 -i shovalmishal/ad-stage1:v3 -e VIT_MODEL_MODE="key" -e LAYER_NUM=11 -e MODEL_TYPE="dino_vit" --pvc=storage:/storage --large-shm --sleep infinity
```

