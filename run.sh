#!/bin/bash
nvcc ./$1.cu -o $1.cuda
read -p "run the program? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
  ./$1.cuda
