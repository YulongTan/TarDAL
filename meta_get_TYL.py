import random
import os

start_number = 0
end_number = 4199  # M3FD 共 4200 对样本

# 所有文件名
all_filenames = [f"{i:05d}.png" for i in range(start_number, end_number + 1)]

# 随机抽取一半（2100）作为训练集
train_filenames = sorted(random.sample(all_filenames, 2100))
val_filenames = sorted(list(set(all_filenames) - set(train_filenames)))

# 写 pred.txt（包含全部）
os.makedirs("meta", exist_ok=True)
with open('meta/pred.txt', 'w') as f:
    for fn in all_filenames:
        f.write(fn + '\n')

# 写 train.txt
with open('meta/train.txt', 'w') as f:
    for fn in train_filenames:
        f.write(fn + '\n')

# 写 val.txt
with open('meta/val.txt', 'w') as f:
    for fn in val_filenames:
        f.write(fn + '\n')

print("✅ meta 文件生成完成！")
