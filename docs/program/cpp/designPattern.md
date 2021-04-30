# 23种设计模式（updating）

## 1 前言

设计模式是软件开发人员在软件开发过程中面临的一般问题的解决方案。这些解决方案是众多软件开发人员经过相当长的一段时间的试验和错误总结出来的。

## 2 UML和时序图

### 2.1 类之间的关系

* 泛化关系 generalization

  继承关系为  is - a，UML图中泛化关系用带空心箭头的直线表示

* 实现关系 realize

  UML图中实现关系用一条带空心箭头的虚线表示

* 聚合关系 aggregation

  UML图中组合关系用一条带空心菱形直线表示

* 组合关系 composition

  UML图中组合关系用一条带实心菱形直线表示

* 关联关系 association

  UML图中关联关系用一条直线表示

* 依赖关系 dependency

  UML图中依赖关系用带箭头的虚线表示

### 2.2 时序图

时序图是显示对象之间交互的图，按照时间顺序排列。

元素：对象、生命线、控制焦点、消息。

## 3 设计模式

根据模式是用来完成什么工作来划分，这种方式可分为5个创建型模式、7个结构型模式和11个行为型模式 3 种。

### 3.1 创建型模式

#### 3.1.1 简单工厂模式

简单工厂模式最大的问题在于工厂类的职责相对过重，增加新的产品需要修改工厂类的判断逻辑，这一点与开闭原则是相违背的。

#### 3.1.2 工厂方法模式

在工厂方法模式中，核心的工厂类不再负责所有产品的创建，而是将具体创建工作交给子类去做。

#### 3.1.3 抽象工厂模式

有时候我们需要一个工厂可以提供多个产品对象，而不是单一的产品对象。

当系统所提供的工厂所需生产的具体产品并不是一个简单的对象，而是多个位于不同产品等级结构中属于不同类型的具体产品时需要使用抽象工厂模式。

#### 3.1.4 建造者模式

建造者模式可以将部件和其组装过程分开，一步一步创建一个复杂的对象。用户只需要指定复杂对象的类型就可以得到该对象，而无须知道其内部的具体构造细节。

#### 3.1.5 单例模式

让类自身负责保存它的唯一实例。这个类可以保证没有其他实例被创建，并且它可以提供一个访问该实例的方法

### 3.2 结构型模式

#### 3.2.1 适配器模式

#### 3.2.2 桥接模式

#### 3.2.3 装饰模式

#### 3.2.4 外观模式

#### 3.2.5 享元模式

#### 3.2.6 代理模式

#### 3.2.7 组合模式

### 3.3 行为型模式

#### 3.3.1 命令模式

#### 3.3.2 中介者模式

#### 3.3.3 观察者模式

#### 3.3.4 状态模式

#### 3.3.5 策略模式

#### 3.3.6 访问者模式

#### 3.3.7 模板模式

#### 3.3.8 备忘录模式

#### 3.3.9 迭代器模式

#### 3.3.10 解释器模式

#### 3.3.11 责任者模式




## 4 参考资料
[1] 图说设计模式
https://design-patterns.readthedocs.io/zh_CN/latest/
