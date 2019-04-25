#!/bin/bash -eux


sudo docker run -it --rm --runtime nvidia -v ${PWD}:/home/app --name mvc-drl-nnabla takuseno/mvc-drl-nnabla bash
