# A quick calculator: how big is a model at each precision?

def model_memory(num_params_billions, byte_per_param):
    return num_params_billions * 1e9 * byte_per_param / (1024**3)  #in GB

for size in [0.5, 7, 13, 70]:
    print(f"\n{size}B parameter model:")
    for name, b in [("FP32", 4), ("FP16", 2), ("INT8", 1), ("INT4", 0.5)]:
        print(f"   {name:5s}: {model_memory(size, b):7.2f} GB")