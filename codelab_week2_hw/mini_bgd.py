# coding=utf-8

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(0., 10., 0.2)
m = len(x)
x0 = np.full(m, 1.0)
input_data = np.vstack([x0,x]).T
target_data = 2 * x + 5 + np.random.rand(m)

# print input_data
# print target_data

loop_max = 1000
epsilon = 1e-3


np.random.seed(0)  #每次生成相同的随机数
w = np.random.randn(2)  #正太分布随机数

alpha = 0.001
diff = 0.
error = np.zeros(2)
count = 0
finish = 0
error_list = []
# 定义batch的大小
batch_size = 5



while count < loop_max:
    count += 1
    for k in range(batch_size):
        a = (m / batch_size) * k
        b = (m / batch_size) * (k + 1)
        sum_m = np.zeros(2)
        for i in range(a,b):
            diff = (np.dot(w, input_data[i]) - target_data[i]) * input_data[i]
            sum_m = sum_m + diff
        w = w - alpha * sum_m
        error_list.append(np.sum(sum_m) ** 2)
    if np.linalg.norm(w - error) < epsilon:
        finish = 1
        break
    else:
        error = w

print 'loop count = %d' % count, '\tw:[%f, %f]' % (w[0], w[1])

#----------------------------------------------------------------------------------------------------------------------


# check with scipy linear regression
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
print 'intercept = %s slope = %s' % (intercept, slope)

plt.plot(range(len(error_list[0:100])), error_list[0:100])
plt.show()

plt.plot(x, target_data, 'k+')
plt.plot(x, w[1] * x + w[0], 'r')
plt.show()



