### 1.TrainingArgment参数详解 
这段说明是关于 `TrainingArguments` 类的使用说明，该类用于定义与训练循环本身相关的参数。通过使用 `HfArgumentParser`，可以将这个类转换成可以在命令行上指定的 `argparse` 参数。
- **output_dir (`str`)**: 模型预测和检查点将被写入的输出目录。
- **overwrite_output_dir (`bool`, 可选，默认为 `False`)**: 如果为 `True`，将覆盖输出目录的内容。如果 `output_dir` 指向一个检查点目录，使用此选项可继续训练。
- **do_train (`bool`, 可选，默认为 `False`)**: 是否执行训练。此参数不直接由 `Trainer` 使用，而是打算由您的训练/评估脚本使用。有关更多详细信息，请查看[示例脚本](https://github.com/huggingface/transformers/tree/main/examples)。
- **do_eval (`bool`, 可选)**: 是否在验证集上执行评估。如果 `evaluation_strategy` 不是 `"no"`，则将设置为 `True`。此参数不直接由 `Trainer` 使用，而是打算由您的训练/评估脚本使用。
- **do_predict (`bool`, 可选，默认为 `False`)**: 是否在测试集上运行预测。此参数不直接由 `Trainer` 使用，而是打算由您的训练/评估脚本使用。
- **evaluation_strategy (`str` 或 `~trainer_utils.IntervalStrategy`, 可选，默认为 `"no"`)**: 在训练期间采用的评估策略。可能的值包括：
  - `"no"`: 训练期间不进行评估。
  - `"steps"`: 每隔 `eval_steps` 进行一次评估（并记录）。
  - `"epoch"`: 在每个epoch结束时进行评估。
- **prediction_loss_only (`bool`, 可选，默认为 `False`)**: 在执行评估和生成预测时，只返回损失。
- **per_device_train_batch_size (`int`, 可选，默认为 8)**: 每个 GPU/XPU/TPU/MPS/NPU 核心/CPU 的训练批量大小。
- **per_device_eval_batch_size (`int`, 可选，默认为 8)**: 每个 GPU/XPU/TPU/MPS/NPU 核心/CPU 的评估批量大小。
- **gradient_accumulation_steps (`int`, 可选，默认为 1)**: 累积梯度的更新步骤数，之后执行向后/更新传递。
- **learning_rate (`float`, 可选，默认为 5e-5)**: 用于 `AdamW` 优化器的初始学习率。
- **weight_decay (`float`, 可选，默认为 0)**: 应用的权重衰减（如果不为零）适用于所有层，除了所有偏差和LayerNorm权重在 `AdamW` 优化器中。
- **adam_beta1 (`float`, 可选，默认为 0.9)**: `AdamW` 优化器的 beta1 超参数。
- **adam_beta2 (`float`, 可选，默认为 0.999)**: `AdamW` 优化器的 beta2 超参数。
- **adam_epsilon (`float`, 可选，默认为 1e-8)**: `AdamW` 优化器的 epsilon 超参数。
- **max_grad_norm (`float`, 可选，默认为 1.0)**: 最大梯度范数（用于梯度裁剪）。
- **num_train_epochs(`float`, 可选，默认为 3.0)**: 执行的总训练epoch数。如果不是整数，将执行最后一个时代的小数部分百分比，然后停止训练。

