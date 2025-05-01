import torch

print("CUDA_count:", torch.cuda.device_count())
print("CUDA_name:", torch.cuda.get_device_name(0))

print(torch.cuda.is_available())


"""

def kernel_correlation_rearrange(n: int, input, output):

	intIndex = (blockIdx.x * blockDim.x) + threadIdx.x

	if (intIndex >= n): return

	intSample = blockIdx.z
	intChannel = blockIdx.y

	fltValue = input[((intSample*input[1] + intChannel) * input[2] * input[3]) + intIndex]

	intPaddedY = (intIndex / input[3]) + 3*1
	intPaddedX = (intIndex % input[3]) + 3*1
	intRearrange = ((input[3] + 6*1) * intPaddedY) + intPaddedX

	output[(((intSample * output[1] * output[2]) + intRearrange) * input[1]) + intChannel] = fltValue

"""


def cpu_launch(fn):
    def launcher(*, grid, block, args):
        grid_x, grid_y, grid_z = grid
        block_x, block_y, block_z = block

        n, input_tensor, output_tensor = args

        for blockIdx_z in range(grid_z):
            for blockIdx_y in range(grid_y):
                for blockIdx_x in range(grid_x):
                    for threadIdx_x in range(block_x):
                        # Создаём контексты blockIdx и threadIdx как объекты
                        blockIdx = type("BlockIdx", (), {
                            "x": blockIdx_x, "y": blockIdx_y, "z": blockIdx_z
                        })()
                        threadIdx = type("ThreadIdx", (), {
                            "x": threadIdx_x
                        })()

                        # Вызываем функцию с созданным контекстом
                        fn(n, input_tensor, output_tensor, blockIdx, threadIdx)
    return launcher
