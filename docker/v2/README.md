# Docker instructions
# ver 2 aims to compare different vit attentions based methods
1. Build the docker:
```shell
docker build -t ad-stage1:v2 -f docker/v2/Dockerfile .
docker tag ad-stage1:v2 shovalmishal/ad-stage1:v2
docker push shovalmishal/ad-stage1:v2
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/workspace/ -e EXPORT_SCRIPT=/workspace/scripts/what_to_say.sh  -e RUN_SCRIPT=/workspace/scripts/say_hello.sh  -it shovalmishal/ad-stage1:v1


```
On runai:
```shell
runai submit --name dino-vit-base-attention-head12 -g 1.0 -i shovalmishal/ad-stage1:v2 -e VIT_MODEL_MODE="class_token_self_attention" -e ATTENTION_NUM=12 -e MODEL_TYPE="dino_vit" -e VIT_ARCH="vit_base" --pvc=storage:/storage --large-shm
#runai submit --name amir-autorep-end2end -g 1.0 -i shovalmishal/ad-stage1:v1 -e EXPORT_SCRIPT=/storage/jevnisek/autorep/scripts/say_hello.sh -e RUN_SCRIPT=/storage/jevnisek/autorep/scripts/run_resnet18_cifar100_end_to_end.sh --pvc=storage:/storage --large-shm
```

