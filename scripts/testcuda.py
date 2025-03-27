# import time
# import torch

# A = torch.ones(5000, 5000).to('cuda')
# B = torch.ones(5000, 5000).to('cuda')
# startTime2 = time.time()
# for i in range(100):
#     C = torch.matmul(A, B)
# endTime2 = time.time()
# print('gpu计算总时长:', round((endTime2 - startTime2) * 1000, 2), 'ms')



# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# import tensorflow as tf

# print('GPU',tf.test.is_gpu_available())

# a = tf.constant(2.)
# b = tf.constant(4.)

# print(a * b)


import tensorrt as trt
print(trt.__version__)
