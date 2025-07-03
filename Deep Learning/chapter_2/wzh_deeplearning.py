import os
import torch
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class EarlyStopping:

    """早停类"""
    def __init__(self, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): 验证集准确率多少个epoch未提升时，进行早停
            verbose (bool): 是否打印早停信息
            delta (float): 判定准确率是否提升的阈值
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            return

        if score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print('EarlyStopping: val_acc did not improve')
        else:
            self.best_score = score
            self.counter = 0


class ModelSaver:
    def __init__(self, model, save_dir='model_weight', save_best_only=True):
        """
        初始化模型保存器
        Args:
            model: 需要保存的模型
            save_path: 模型保存路径
            save_best_only: 是否只保存最佳模型
        """
        self.model = model
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.best_score = None  # 用于记录最佳准确率
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  # 创建保存目录
        
    def __call__(self, model, epoch, val_acc):
        """
        保存模型
        
        参数:
            model: 需要保存的模型
            epoch: 当前训练轮数
            val_acc: 当前验证准确率
        """
        # 生成文件名
        filename = f'model_epoch_{epoch}_acc_{val_acc:.4f}.pth'
        save_path = os.path.join(self.save_dir, filename)
        
        # 是否仅保存最佳模型
        if self.save_best_only:
            # 首次调用或验证准确率提高时保存
            if self.best_score is None or val_acc > self.best_score:
                self.best_score = val_acc
                torch.save(model.state_dict(), save_path)
                
                # 删除之前的最佳模型
                for old_file in os.listdir(self.save_dir):
                    if old_file != filename and old_file.endswith('.pth'):
                        os.remove(os.path.join(self.save_dir, old_file))
        else:
            # 每个epoch都保存
            torch.save(model.state_dict(), save_path)


def classification_evaluate(model, data_loader, criterion, device):
    """评估模型在数据集上的表现"""
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():  # 不计算梯度
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_classification(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    num_epochs=10, 
    model_saver=None,
    early_stopping=None,
    eval_step=500
):
    """
    基于tqdm的训练函数，与training函数类似
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮次
        model_saver: 保存检查点回调函数
        early_stopping: 早停回调函数
        eval_step: 每多少步评估一次
    
    返回:
        record_dict: 包含训练和验证记录的字典
    """
    record_dict = {
        "train": [],
        "val": []
    }
    
    global_step = 0
    model.train()
    with tqdm(total=num_epochs * len(train_loader)) as pbar:
        for epoch_id in range(num_epochs):
            # 训练
            for datas, labels in train_loader:
                datas = datas.to(device)  # 数据放到device上
                labels = labels.to(device)  # 标签放到device上
                
                # 梯度清空
                optimizer.zero_grad()
                
                # 模型前向计算
                logits = model(datas)
                
                # 计算损失
                loss = criterion(logits, labels)
                
                # 梯度回传，计算梯度
                loss.backward()
                
                # 更新模型参数
                optimizer.step()
                
                # 计算准确率
                preds = logits.argmax(axis=-1)
                acc = (preds == labels).float().mean().item() * 100
                loss_value = loss.cpu().item()
                
                # 记录训练数据
                record_dict["train"].append({
                    "loss": loss_value, "acc": acc, "step": global_step
                })
                
                # 评估
                if global_step % eval_step == 0:
                    val_loss, val_acc = classification_evaluate(model, val_loader, criterion, device)
                    record_dict["val"].append({
                        "loss": val_loss, "acc": val_acc, "step": global_step
                    })
                    model.train()  # 切换回训练集模式
            
                    # 保存模型权重
                    # 如果有模型保存器，保存模型
                    if model_saver is not None:
                        model_saver(model, val_acc, epoch_id)
                    
                    # 如果有早停器，检查是否应该早停
                    if early_stopping is not None:
                        early_stopping(val_acc)
                        if early_stopping.early_stop:
                            print(f'早停: 已有{early_stopping.patience}轮验证损失没有改善！')
                            return model, record_dict
                            
                # 更新步骤
                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"epoch": epoch_id, "loss": f"{loss_value:.4f}", 
                                  "acc": f"{acc:.2f}%"})
    
    return model, record_dict


# 画线要注意的是损失是不一定在零到1之间的
def plot_learning_curves(record_dict, sample_step=500):  # sample_step表示每隔多少个step取一个值
    # build DataFrame
    train_df = pd.DataFrame(record_dict["train"]).set_index("step").iloc[::sample_step]  # 每隔500个step取一个值
    val_df = pd.DataFrame(record_dict["val"]).set_index("step")
    # print(train_df.head())
    # print(val_df.head())
    # plot
    fig_num = len(train_df.columns)  # 因为有loss和acc两个指标，所以画个子图
    fig, axs = plt.subplots(1, fig_num, figsize=(5 * fig_num, 5))  # fig_num个子图，figsize是子图大小
    
    # 设置总标题
    fig.suptitle('Training and Validation Metrics', fontsize=16)
    
    for idx, item in enumerate(train_df.columns):    
        # index是步数，item是指标名字
        axs[idx].plot(train_df.index, train_df[item], label=f"train_{item}")
        axs[idx].plot(val_df.index, val_df[item], label=f"val_{item}")
        axs[idx].grid()
        axs[idx].legend()
        x_data = range(0, train_df.index[-1], 5000)  # 每隔5000步标出一个点
        axs[idx].set_xticks(x_data)
        axs[idx].set_xticklabels(map(lambda x: f"{int(x/1000)}k", x_data))  # map生成labal
        axs[idx].set_xlabel("step")
        # 设置每个子图的标题
        axs[idx].set_title(f'{item.capitalize()} over Training Steps')
        # 设置y轴从0开始
        axs[idx].set_ylim(bottom=0)
    
    plt.show()


def plot_learning_loss_curves(record_dict, sample_step=500):
    """
    画学习曲线，横坐标是steps，纵坐标是loss和acc,回归问题只有loss

    参数:
        record_dict: 包含训练和验证记录的字典
        sample_step: 每多少步画一个点，默认500步
    """
    train_df = pd.DataFrame(record_dict["train"]).set_index("step").iloc[::sample_step]
    val_df = pd.DataFrame(record_dict["val"]).set_index("step")

    # 只绘制一个loss图，不需要循环
    fig, ax = plt.subplots(figsize=(10, 5))  # 创建单个图表

    # 绘制训练和验证的loss曲线
    ax.plot(train_df.index, train_df['loss'], label="train_loss")
    ax.plot(val_df.index, val_df['loss'], label="val_loss")

    # 设置图表属性
    ax.grid()
    ax.legend()
    x_data = range(0, train_df.index[-1], 5000)  # 每隔5000步标出一个点
    ax.set_xticks(x_data)
    ax.set_xticklabels(map(lambda x: f"{int(x / 1000)}k", x_data))  # map生成label
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("loss curves")

    plt.show()


def train_regression_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device='cpu', 
    num_epochs=100, 
    print_every=10,
    eval_step=500,
    model_saver=None,
    early_stopping=None
):
    """
    训练回归模型的函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮次
        print_every: 每多少轮打印一次结果
        eval_step: 每多少步评估一次
    
    返回:
        record_dict: 包含训练和验证记录的字典
    """
    record_dict = {
        "train": [],
        "val": []
    }
    
    global_step = 0
    model.train()
    epoch_val_loss = 0
    
    with tqdm(total=num_epochs * len(train_loader), desc="train progress") as pbar:
        for epoch_id in range(num_epochs):
            # 训练
            model.train()
            running_loss = 0.0
            
            for datas, labels in train_loader:
                # 将标签放到device上
                targets = labels.to(device)
                
                # 处理输入数据
                if isinstance(datas, list) or isinstance(datas, tuple):
                    # 如果datas是list或tuple，则将每个元素都放到device上
                    inputs = [x.to(device) for x in datas]
                    # 使用解包操作将多个输入传递给模型
                    outputs = model(*inputs)
                else:
                    inputs = datas.to(device)
                    outputs = model(inputs)
                
                # 梯度清空
                optimizer.zero_grad()
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 梯度回传，计算梯度
                loss.backward()
                
                # 更新模型参数
                optimizer.step()
                
                # 更新步骤
                global_step += 1
            
                # 在每个批次后记录训练损失
                epoch_train_loss = loss.item()
                record_dict["train"].append({
                    "loss": epoch_train_loss,
                    "step": global_step
                })
                
                # 验证
                if global_step % eval_step == 0:
                    epoch_val_loss = evaluate_regression_model(model, val_loader, device, criterion)
            
                    # 记录验证数据
                    record_dict["val"].append({
                        "loss": epoch_val_loss, 
                        "step": global_step
                    })
                    # 保存模型权重
                    # 如果有模型保存器，保存模型
                    if model_saver is not None:
                        model_saver(model, -epoch_val_loss, epoch_id)
                    
                    # 如果有早停器，检查是否应该早停
                    if early_stopping is not None:
                        early_stopping(-epoch_val_loss)
                        if early_stopping.early_stop:
                            print(f'早停: 已有{early_stopping.patience}轮验证损失没有改善！')
                            return model, record_dict
            
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"loss": f"{epoch_train_loss:.4f}", 
                                  "val_loss": f"{epoch_val_loss:.4f}, global_step{global_step}"})
    
    return model, record_dict


# 评估模型
def evaluate_regression_model(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():  # 禁止 autograd 记录计算图，节省显存与算力。
        for inputs, targets in dataloader:
            targets = targets.to(device)
            if isinstance(inputs, list) or isinstance(inputs, tuple):  # 如果inputs是list或tuple，则将每个元素都放到device上
                inputs = [x.to(device) for x in inputs]
                outputs = model(*inputs)  # 使用解包操作将多个输入传递给模型
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)  # 前向计算
            
            loss = criterion(outputs, targets)  # 计算损失
            
            running_loss += loss.item() * targets.size(0)
    
    return running_loss / len(dataloader.dataset)


def train_multi_output_regression_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device='cpu', 
    num_epochs=100, 
    print_every=10,
    eval_step=500,
    model_saver=None,
    early_stopping=None
):
    """
    训练回归模型的函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮次
        print_every: 每多少轮打印一次结果
        eval_step: 每多少步评估一次
    
    返回:
        record_dict: 包含训练和验证记录的字典
    """
    record_dict = {
        "train": [],
        "val": []
    }
    
    global_step = 0
    model.train()
    epoch_val_loss = 0
    
    with tqdm(total=num_epochs * len(train_loader), desc="train progress") as pbar:
        for epoch_id in range(num_epochs):
            # 训练
            model.train()
            running_loss = 0.0
            
            for datas, labels in train_loader:
                # 假设inputs是一个包含多个tensor的元组，targets是最后一个元素
                targets = labels.to(device)
                if isinstance(datas, tuple) or isinstance(datas, list):  # 如果datas是tuple或list，则将每个元素都放到device上
                    inputs = [x.to(device) for x in datas]
                else:
                    inputs = datas.to(device)
                
                # 梯度清空
                optimizer.zero_grad()
                
                # 模型前向计算
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    # 检查inputs的长度
                    if len(inputs) == 2:
                        # 如果inputs包含两个元素，则将第一个元素传递给模型
                        # 这是针对HousingDataset.__getitem__返回(self.features[idx],self.features[idx][-2:])的情况
                        outputs = model(inputs[0])
                    else:
                        # 否则使用解包操作将多个输入传递给模型
                        outputs = model(*inputs)
                else:
                    outputs = model(inputs)
                # 计算损失
                output, deep = outputs
                # 处理deep：求平均，reshape为output尺寸，并和output相加
                deep_mean = torch.mean(deep, dim=1)  # 沿着第1维求平均
                deep_reshaped = deep_mean.view_as(output)  # 重塑为与output相同的尺寸
                # 分别计算output和deep_reshaped的损失，然后求和
                loss_output = criterion(output, targets)
                loss_deep = criterion(deep_reshaped, targets)
                loss = loss_output + loss_deep  # 总损失为两部分之和
                
                # 梯度回传，计算梯度
                loss.backward()
                
                # 更新模型参数
                optimizer.step()
                
                # 更新步骤
                global_step += 1
            
                # 在每个批次后记录训练损失
                epoch_train_loss = loss.item()
                record_dict["train"].append({
                    "loss": epoch_train_loss,
                    "step": global_step
                })
                
                # 验证
                if global_step % eval_step == 0:
                    epoch_val_loss = evaluate_multi_output_regression_model(model, val_loader, device, criterion)
            
                    # 记录验证数据
                    record_dict["val"].append({
                        "loss": epoch_val_loss, 
                        "step": global_step
                    })
                    # 保存模型权重
                    # 如果有模型保存器，保存模型
                    if model_saver is not None:
                        model_saver(model, -epoch_val_loss, epoch_id)
                    
                    # 如果有早停器，检查是否应该早停
                    if early_stopping is not None:
                        early_stopping(-epoch_val_loss)
                        if early_stopping.early_stop:
                            print(f'早停: 已有{early_stopping.patience}轮验证损失没有改善！')
                            return model, record_dict
            
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"loss": f"{epoch_train_loss:.4f}", 
                                  "val_loss": f"{epoch_val_loss:.4f}, global_step{global_step}"})
    
    return model, record_dict


# 评估模型
def evaluate_multi_output_regression_model(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():  # 禁止 autograd 记录计算图，节省显存与算力。
        for inputs, targets in dataloader:
            targets = targets.to(device)
            if isinstance(inputs, tuple) or isinstance(inputs, list):  # 如果inputs是tuple或list，则将每个元素都放到device上
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)

            # 模型前向计算
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                # 检查inputs的长度
                if len(inputs) == 2:
                    # 如果inputs包含两个元素，则将第一个元素传递给模型
                    # 这是针对HousingDataset.__getitem__返回(self.features[idx],self.features[idx][-2:])的情况
                    outputs = model(inputs[0])
                else:
                    # 否则使用解包操作将多个输入传递给模型
                    outputs = model(*inputs)
            else:
                outputs = model(inputs)
            output, deep = outputs
            # 处理deep：求平均，reshape为output尺寸，并和output相加
            deep_mean = torch.mean(deep, dim=1)  # 沿着第1维求平均
            deep_reshaped = deep_mean.view_as(output)  # 重塑为与output相同的尺寸
            loss_output = criterion(output, targets)
            loss_deep = criterion(deep_reshaped, targets)
            loss = loss_output + loss_deep  # 总损失为两部分之和
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)