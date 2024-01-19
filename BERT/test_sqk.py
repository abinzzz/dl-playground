from scipy.special import softmax  
import numpy as np  

def test_gradient(dimension, time_steps=50, scaling_factor=1.0):
    """
    参数：
    - dimension: 查询向量和键向量的维度。
    - time_steps: 生成键向量的数量。
    - scaling_factor: 应用于点积的缩放因子。

    返回值：
    - 梯度矩阵中最大的绝对值分量。
    """

    # 生成随机的查询向量和键向量，其组成部分从标准正态分布中抽取
    query_vector = np.random.randn(dimension)
    key_vectors = np.random.randn(time_steps, dimension)

    # 计算点积，应用缩放，然后计算softmax
    dot_products = np.sum(query_vector * key_vectors, axis=1) / scaling_factor
    softmax_output = softmax(dot_products)

    # 计算softmax输出的梯度
    gradient_matrix = np.diag(softmax_output) - np.outer(softmax_output, softmax_output)

    # 返回梯度矩阵中的最大绝对值
    return np.max(np.abs(gradient_matrix))

# 实验次数
NUMBER_OF_EXPERIMENTS = 5

# 运行没有缩放的实验
results_without_scaling_100 = [test_gradient(100) for _ in range(NUMBER_OF_EXPERIMENTS)]
results_without_scaling_1000 = [test_gradient(1000) for _ in range(NUMBER_OF_EXPERIMENTS)]

# 运行有缩放的实验
results_with_scaling_100 = [test_gradient(100, scaling_factor=np.sqrt(100)) for _ in range(NUMBER_OF_EXPERIMENTS)]
results_with_scaling_1000 = [test_gradient(1000, scaling_factor=np.sqrt(1000)) for _ in range(NUMBER_OF_EXPERIMENTS)]

# 打印结果
print("没有缩放的结果（维度=100）:", results_without_scaling_100)
print("没有缩放的结果（维度=1000）:", results_without_scaling_1000)
print("有缩放的结果（维度=100）:", results_with_scaling_100)
print("有缩放的结果（维度=1000）:", results_with_scaling_1000)


import matplotlib.pyplot as plt
# Revising the plot to ensure all data points are included

# Given experimental data with correct values
data = {
    "No Scaling dim=100": [0.059398546712975064, 0.2498360169388831, 0.008179271245615127, 0.16985040166173004, 0.00017518204173572194],
    "No Scaling dim=1000": [0.037403200843576845, 1.8829382497642655e-11, 6.995490600791854e-06, 1.3460521586239338e-10, 5.498179689311655e-11],
    "With Scaling dim=100": [0.23435524441068933, 0.10572976561186455, 0.09711877538913292, 0.059005529454577245, 0.15737320167534957],
    "With Scaling dim=1000": [0.12238213059896091, 0.09907377893252199, 0.09771834771001327, 0.08899382001739972, 0.1312868174831885]
}

# Recreate the combined scatter plot for all datasets
plt.figure(figsize=(10, 6))

# Assign colors and labels to each dataset
colors = ['red', 'blue', 'green', 'purple']
labels = ['No Scaling dim=100', 'No Scaling dim=1000', 'With Scaling dim=100', 'With Scaling dim=1000']
markers = ['o', '^', 's', 'x']

# Plot each dataset with a unique color and label
for i, (label, results) in enumerate(data.items()):
    x_values = [i + 1] * len(results)  # create x values to spread the points horizontally
    plt.scatter(x_values, results, color=colors[i], label=labels[i], marker=markers[i], alpha=0.7)

# Set the chart title and labels
plt.title('Comparison of Max Gradient Absolute Values Across Different Conditions')
plt.xlabel('Experiment Condition')
plt.ylabel('Max Gradient Absolute Value')

# Add a legend to the plot
plt.legend()

# Adjust layout for better fit
plt.tight_layout()

# Show the scatter plot
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# 给定的实验数据
data = {
    "No Scaling dim=100": [0.059398546712975064, 0.2498360169388831, 0.008179271245615127, 0.16985040166173004, 0.00017518204173572194],
    "No Scaling dim=1000": [0.037403200843576845, 1.8829382497642655e-11, 6.995490600791854e-06, 1.3460521586239338e-10, 5.498179689311655e-11],
    "With Scaling dim=100": [0.23435524441068933, 0.10572976561186455, 0.09711877538913292, 0.059005529454577245, 0.15737320167534957],
    "With Scaling dim=1000": [0.12238213059896091, 0.09907377893252199, 0.09771834771001327, 0.08899382001739972, 0.1312868174831885]
}

# 创建图形
plt.figure(figsize=(10, 6))

# 为每个数据集分配颜色和标签
colors = ['red', 'blue', 'green', 'purple']
labels = ['No Scaling dim=100', 'No Scaling dim=1000', 'With Scaling dim=100', 'With Scaling dim=1000']
markers = ['o', '^', 's', 'x']

# 对每个数据集进行绘图，使用唯一的颜色和标签
for i, (label, results) in enumerate(data.items()):
    # 为了避免对数变换问题，对所有数值加上一个小正数
    adjusted_results = np.array(results) + 1e-12
    x_values = [i + 1] * len(results)  # 创建x值以水平分散点
    plt.scatter(x_values, np.log10(adjusted_results), color=colors[i], label=labels[i], marker=markers[i], alpha=0.7)

# 设置图表标题和标签
plt.title('Comparison of Max Gradient Absolute Values Across Different Conditions')
plt.xlabel('Experiment Condition')
plt.ylabel('Log10 of Max Gradient Absolute Value')

# 添加图例
plt.legend()

# 调整布局以更好地适应
plt.tight_layout()

# 显示散点图
plt.show()