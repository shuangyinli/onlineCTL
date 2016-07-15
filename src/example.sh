#!/bin/bash

#This is simple example how to use online CTL for training.

#The train set is the splits of IMDB which contains more than 200,000 movies.
#
#Check ../demo/ to show the input split files:
#Check ./output to show the output 
# you must modify the setting.txt before you run your data.


make clean
echo
make
echo
rm -f ./output/*

echo

time ./ctl est ../demo/ setting.txt 20 ./output

echo
