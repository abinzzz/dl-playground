### 1.继承是什么？
如下定义一个动物类Animal为基类，它基本两个实例属性name和age、一个方法call。

```python
class Animal(object):  #  python3中所有类都可以继承于object基类
   def __init__(self, name, age):
       self.name = name
       self.age = age

   def call(self):
       print(self.name, '会叫')

######
# 现在我们需要定义一个Cat 猫类继承于Animal，猫类比动物类多一个sex属性。 
######
class Cat(Animal):
   def __init__(self,name,age,sex):
       super(Cat, self).__init__(name,age)  # 不要忘记从Animal类引入属性
       self.sex=sex

if __name__ == '__main__':  # 单模块被引用时下面代码不会受影响，用于调试
   c = Cat('喵喵', 2, '男')  #  Cat继承了父类Animal的属性
   c.call()  # 输出 喵喵 会叫 ，Cat继承了父类Animal的方法 
```  
注意：一定要用 super(Cat, self).__init__(name,age) 去初始化父类，否则，继承自 Animal的 Cat子类将没有 name和age两个属性。

函数super(Cat, self)将返回当前类继承的父类，即 Animal，然后调用__init__()方法，注意self参数已在super()中传入，在__init__()中将隐式传递，不能再写出self。

注意当`call`函数没有被定义的时候，会使用父类的`call`函数，而子类已经定义`call`函数的时候会优先使用自身定义的`call`函数。




### 2. `nn.Moudle`详解
```python

class Module(object):
    def __init__(self):
    def forward(self, *input):
 
    def add_module(self, name, module):
    def cuda(self, device=None):
    def cpu(self):
    def __call__(self, *input, **kwargs):
    def parameters(self, recurse=True):
    def named_parameters(self, prefix='', recurse=True):
    def children(self):
    def named_children(self):
    def modules(self):  
    def named_modules(self, memo=None, prefix=''):
    def train(self, mode=True):
    def eval(self):
    def zero_grad(self):
    def __repr__(self):
'''
有一部分没有完全列出来
'''
```

解释：
- training (bool) - 指示模块当前是训练还是评估模式
- add_module() - 添加子模块
- apply() - 递归地将函数应用于所有子模块
- buffers() - 返回模块 buffer 的迭代器 
- children() - 返回直接子模块的迭代器
- cpu()/cuda()/etc. - 将模块移动到相应设备
- double()/float()/etc. - 将模块参数和 buffer 转换为相应数据类型
- eval() - 将模块设为评估模式
- forward() - 定义前向传播计算,所有子类需要重写
- register_buffer() - 向模块添加 buffer
- register_parameter() - 向模块添加参数
- state_dict() - 返回模块状态的字典表示
- load_state_dict() - 从字典中加载模块状态
- parameters()/named_parameters() - 返回可训练参数的迭代器
- modules()/named_modules() - 返回所有子模块的迭代器
- zero_grad() - 将所有参数的梯度设为0
- train()/eval() - 设置模块为训练/评估模式

### 3.注意技巧
我们一般定义自己的网络的时候，会继承这个`nn.Moudle`,并重新构造`__init__`和`forward`这两个def，但有一些技巧需要注意：
- 将具有**可学习参数的层**放在构造函数`__init__`中
- foward方法必须重写，实现各个层连接


```python

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu1=torch.nn.ReLU()
        self.max_pooling1=torch.nn.MaxPool2d(2,1)
 
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu2=torch.nn.ReLU()
        self.max_pooling2=torch.nn.MaxPool2d(2,1)
 
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pooling2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
 
model = MyNet()
print(model)
'''运行结果为：
MyNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (max_pooling1): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu2): ReLU()
  (max_pooling2): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)
'''
```




### 参考链接
- [pytorch教程之nn.Module类详解——使用Module类来自定义模型](https://blog.csdn.net/qq_27825451/article/details/90550890)
- [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)