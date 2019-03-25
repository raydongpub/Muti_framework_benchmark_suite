srun -p gpu3 -N 1 -n 20 -t 0-05:00 --mem=100G -w lewis4-r730-gpu3-node429 --gres=gpu:1 --exclusive --pty /bin/bash
