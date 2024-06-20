# Docker instructions
1. Build the docker:
```shell
docker build -t ad-stage1:base -f docker/base/Dockerfile .
docker tag ad-stage1:base shovalmishal/ad-stage1:base
docker push shovalmishal/ad-stage1:base
```
