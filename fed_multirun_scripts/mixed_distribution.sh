python MoDL_FedAvg_train.py --GPU 0 --num_work 7 --ch 16 --num_pool 3 --client_pats 5 5 5 5 5 5 5 5 5 5 --client_sites 1 2 3 4 5 6 7 8 9 10 --share_int 50 --train_dilation 8 &
python MoDL_FedAvg_train.py --GPU 1 --num_work 7 --ch 16 --num_pool 3 --client_pats 5 5 5 5 5 5 5 5 5 5 --client_sites 1 2 3 4 5 6 7 8 9 10 --share_int 100 --train_dilation 4 &
python MoDL_FedAvg_train.py --GPU 2 --num_work 7 --ch 16 --num_pool 3 --client_pats 5 5 5 5 5 5 5 5 5 5 --client_sites 1 2 3 4 5 6 7 8 9 10 --share_int 20 --train_dilation 20 &
python MoDL_FedAvg_train.py --GPU 3 --num_work 7 --ch 16 --num_pool 3 --client_pats 5 5 5 5 5 5 5 5 5 5 --client_sites 1 2 3 4 5 6 7 8 9 10 --share_int 10 --train_dilation 40 &
python MoDL_FedAvg_train.py --GPU 4 --num_work 7 --ch 16 --num_pool 3 --client_pats 5 5 5 5 5 5 --client_sites 4 5 6 7 8 9 --share_int 50 --train_dilation 16 &
python MoDL_FedAvg_train.py --GPU 5 --num_work 7 --ch 16 --num_pool 3 --client_pats 5 5 5 5 5 5 --client_sites 4 5 6 7 8 9 --share_int 100 --train_dilation 8 &
python MoDL_FedAvg_train.py --GPU 6 --num_work 7 --ch 16 --num_pool 3 --client_pats 5 5 5 5 5 5 --client_sites 4 5 6 7 8 9 --share_int 20 --train_dilation 40 &
python MoDL_FedAvg_train.py --GPU 7 --num_work 7 --ch 16 --num_pool 3 --client_pats 5 5 5 5 5 5 --client_sites 4 5 6 7 8 9 --share_int 10 --train_dilation 80 &