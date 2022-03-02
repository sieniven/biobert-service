FROM tensorflow/tensorflow:1.15.5-gpu-py3

ADD . /biobert
WORKDIR /biobert