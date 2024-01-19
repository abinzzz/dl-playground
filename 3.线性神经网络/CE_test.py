import torch

# 定义交叉熵损失函数，对概率取对数后取负
def cross_entropy(y_hat, y):
    # Step 1: Generate a range from 0 to the length of y_hat
    indices = range(len(y_hat))
    print("indices:", indices)

    # Step 2: Select the predicted probabilities of the true labels
    selected_probs = y_hat[indices, y]
    print("selected_probs:", selected_probs)

    # Step 3: Compute the log of the selected probabilities
    log_probs = torch.log(selected_probs)
    print("log_probs:", log_probs)

    # Step 4: Negate the log probabilities
    neg_log_probs = -log_probs
    print("neg_log_probs:", neg_log_probs)

    return neg_log_probs



# 定义预测概率和真实标签
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.tensor([0, 2])

# 计算交叉熵损失
loss = cross_entropy(y_hat, y)

# 打印损失
print(loss)