#!/bin/bash

#!/bin/bash

./nw-cpu-gpu -k 0 -n 1  -l 32 --debug
./nw-cpu-gpu -k 0 -n 10 -l 128 --debug
./nw-cpu-gpu -k 0 -n 10 -f 50 -t 96 -l 512 --debug
./nw-cpu-gpu -k 0 -n 20 -t 96 -b 16 -l 1600 --debug

./nw-cpu-gpu -k 1 -n 1 -l 32 --debug
./nw-cpu-gpu -k 1 -n 10 -l 128 --debug
./nw-cpu-gpu -k 1 -n 10 -l 512 --debug
./nw-cpu-gpu -k 1 -n 20 -b 16 -l 1600 --debug

