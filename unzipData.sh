#!/bin/bash

for name in $(ls ./dataset)
do
    unzip ./dataset/$name -d ./dataset
    rm ./dataset/$name
done