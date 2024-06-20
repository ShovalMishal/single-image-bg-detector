FROM shovalmishal/ad-stage1:base
WORKDIR /home/code/
COPY . ./
CMD ["/bin/bash", "-c", "source $EXPORT_SCRIPT && source $RUN_SCRIPT"]
