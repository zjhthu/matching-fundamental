import torch
from multiprocessing import Pool as ThreadPool 

def tocuda(data):
	# convert tensor data in dictionary to cuda when it is a tensor
	for key in data.keys():
		if type(data[key]) == torch.Tensor:
			data[key] = data[key].cuda()
	return data

def get_pool_result(num_processor, fun, args):
    pool = ThreadPool(num_processor)
    pool_res = pool.map(fun, args)
    pool.close()
    pool.join()
    return pool_res