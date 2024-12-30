import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import queue
import threading
from random import shuffle
from datetime import datetime
import time
from collections import OrderedDict

#全局变量统计
processed_count = 0
learned_count = 0
skip_count = 0
product_count = 0
stop_standard = 100

def thread_end_condition():
    global learned_count
    global stop_standard
    return learned_count>stop_standard

# 数据预处理
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST数据集
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return trainloader

# 定义 Token 数据结构
class Token:
    def __init__(self, id, tensor, timestamp=None):
        self.id = id  # 编号
        self.tensor = tensor  # Tensor 数据
        self.timestamp = timestamp  # 时间戳

# 定义子模型1（提取特征部分）
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        model = models.alexnet(pretrained=True)
        self.features = model.features
        self.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 修改为1通道输入
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))  # 确保输出大小始终为 6x6

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        return x.flatten(start_dim=1)  # 展平以适应Model2

# 定义子模型2（分类部分）
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        model = models.alexnet(pretrained=True)
        self.classifier = model.classifier
        self.classifier[6] = nn.Linear(4096, 10)  # 修改输出层适应MNIST（10个类别）

    def forward(self, x):
        return self.classifier(x)

# 生产者函数 - 将数据集中的数据封装为 Token 并放入队列
def producer(data_loader, task_queue, learn_dict, num_producers=2):
    product_threads = []
    for _ in range(num_producers):
        product_threads.append(threading.Thread(target=_producer_thread, args=(data_loader, task_queue, learn_dict,thread_end_condition)))
    for thread in product_threads:
        thread.start()
    return product_threads
def _producer_thread(data_loader, task_queue, learn_dict,thread_end_func):
    for idx,(input, label) in enumerate(data_loader):  
        timestamp = datetime.now().timestamp()
        token = Token(id=1, tensor=input, timestamp=timestamp)
        task_queue.put(token)
        learn_dict[timestamp] = label
        print('put one sample to the queue')
        global product_count,processed_count 
        product_count+=1
        if product_count>1000+processed_count:
            time.sleep(1)
        # 添加终止条件
        if thread_end_func():
            break

# 消费者函数 - 从队列中提取 Token 进行前向计算
def consumer(model1, model2, device, task_queue, output_queue,learn_dict, num_consumers=2):
    processed_threads = []
    for _ in range(num_consumers):
        processed_threads.append(threading.Thread(target=_consumer_thread, args=(model1, model2, device, task_queue, output_queue,learn_dict,thread_end_condition)))
    
    for thread in processed_threads:
        thread.start()
    return processed_threads

def _consumer_thread(model1, model2, device, task_queue, output_queue,learn_dict,thread_end_func):
    global processed_count
    while True:
        try:
            token = task_queue.get(timeout=1)  # 超时以允许线程退出
        except queue.Empty:
            continue
        if token.id == 1:
            output = model1(token.tensor.to(device))
            token.tensor = output
            token.id = 2
            task_queue.put(token)
        elif token.id == 2:
            outputs = model2(token.tensor.to(device))
            print(f'put one output to the output')
            if token.timestamp in learn_dict:
                output_queue.put([outputs,learn_dict[token.timestamp]])
            else:
                print('error output without label record')
            processed_count+=1
        #制定退出条件
        # 添加终止条件
        if thread_end_func():
            break

# 学习者函数 - 从 output_queue 中提取输出并基于 learn_queue 中的标签进行反向传播
def learner(model1,model2, device, learn_dict,num_learners = 1):
    learner_threads = []
    for _ in range(num_learners):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=0.001)
        learner_threads.append(threading.Thread(target=_learner_offline_thread, args=(optimizer,criterion, device, learn_dict,thread_end_condition)))
    
    for thread in learner_threads:
        thread.start()
    return learner_threads

def _learner_offline_thread(optimizer, criterion, device, output_queue,thread_end_func):
    #离线学习策略:学习标签先于输出产生，且所有输入均会获得反馈
    global skip_count
    global learned_count
    while True:
        try:
            #从可学习的标签中拿取一个标签
            bp_sample = output_queue.get(timeout = 1)
            optimizer.zero_grad()
            loss = criterion(bp_sample[0], bp_sample[1].to(device))
            try:
                loss.backward()
                learned_count+=1
                print('learn task')
            except RuntimeError:
                print('skip this training step')
                skip_count+=1
            optimizer.step()
        except queue.Empty:
            continue
        #制定退出条件
        if thread_end_func():
            break
# 主程序
def main():
    torch.autograd.set_detect_anomaly(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    trainloader = load_data(batch_size=16)

    # 初始化模型、优化器和损失函数
    model1 = Model1().to(device)
    model2 = Model2().to(device)

    # 创建任务队列和输出队列
    task_queue = queue.Queue()
    output_queue = queue.Queue()
    learn_dict = {}

    # 启动生产者、消费者和学习者线程
    product_threads = producer(trainloader, task_queue, learn_dict)
    processed_threads = consumer(model1, model2, device, task_queue, output_queue,learn_dict)
    learner_threads = learner(model1,model2,device,output_queue)

    for thread in processed_threads:
            thread.join()
    for thread in product_threads:
        thread.join()
    for thread in learner_threads:
        thread.join()
    global learned_count
    global processed_count
    global learned_count
    global skip_count
    print(f"product count {product_count}")
    print(f"processed count {processed_count}")
    print(f"learned count {learned_count}")
    print(f"skip count {skip_count}")
        
        

    # 在实际应用中，你需要确保有方法通知线程停止工作并正确退出。

if __name__ == "__main__":
    main()