#nohup python3 bareplate.py --output 'data' --data '../data/bareplate/data.npy' --data_num 100000 --iters 50000 --device '0'
#nohup python3 bareplate.py --output 'datanoisy' --data '../data/bareplate/data_noisy.npy' --data_num 100000 --iters 50000 --device '0'
#nohup python3 bareplate.py --output 'invers' --data '../data/bareplate/data_partial.npy' --data_num 0 --iters 50000 --calib_eta True --device '0'
#nohup python3 bareplate.py --output 'inverseta' --data '../data/bareplate/data_partial_noisy.npy' --data_num 0 --iters 50000 --calib_eta True --device '0'
# nohup python3 bareplate.py --output 'inversmat' --data '../data/bareplate/data_partial_noisy.npy' --data_num 0 --iters 50000 --calib_material True --device '1'
#nohup python3 bareplate.py --output 'nodata' --iters 50000  --device '0'
#nohup python3 bareplate.py --output 'transfer' --iters 10000 --device '1' --valid '../data/bareplate/data_400W.npy' --pretrain '../results/bareplate/nodata.pt' --v 8. --P 400. --t_end 3.75
#nohup python3 bareplate.py --output 'notransfer' --iters 50000 --device '1' --valid '../data/bareplate/data_400W.npy' --v 8. --P 400. --t_end 3.75
nohup python3 2Dwall.py --iters 100000 
nohup python3 2Dwall.py --iters 5000 --task 'calibration'