---
title: SGLang 并行策略分析：一组 GPU，究竟该怎么拆？
description: 从切分对象、通信代价和显存布局出发，分析 SGLang 的 TP、DP、PP、EP、CP/DCP 与 PD 分离，并用 8 卡示例解释组合关系和选型边界。
date: 2026-09-18
---

# SGLang 并行策略分析：一组 GPU，究竟该怎么拆？

同样是 8 张 GPU，为什么有时应该做 TP=8，有时应该拆成两个 TP=4 的副本，而在 MoE 模型上，又会看到 TP、DP Attention、EP 同时开启？

问题的关键在于：**我们究竟在切分什么，以及为了这次切分，需要付出多少通信代价。**

TP 切张量，PP 切模型层，普通 DP 分请求，EP 分专家，CP 分上下文。DP Attention 和 Prefill/Decode 分离又分别改变了层内计算与服务阶段的组织方式。名字都带“并行”，资源计算和性能收益却不能混为一谈。

本文围绕 SGLang 的自回归大模型推理，分析这些策略的工作方式、组合关系与选型边界。多模态只讨论与服务拆分相关的扩展，不展开扩散模型并行。

> 版本说明：文中参数与实现按 2026 年 9 月 18 日核对的 SGLang 官方文档、上游源码快照 `1f60ddef5dc2` 说明。配置数字用于解释拓扑，不代表任意模型都能直接运行；本文未进行 GPU 性能实测。

## 01｜先分清：切模型，还是分请求？

想理解并行策略，可以先看三个基本问题。

**模型装不下时，需要分摊权重。** 可以把权重拆到多张卡上。TP 按张量维度拆分，PP 按层拆分；MoE 模型还可以用 EP 分配专家。

**一个实例忙不过来时，可以增加副本。** 把不同请求分给不同实例。这是普通 DP 的主要作用。

**一条长请求太重时，可以拆分上下文。** 让多张卡协同处理它。Prefill CP 与 Decode Context Parallelism（DCP）分别面向不同阶段组织这种协作。

![TP、PP、DP 分别切分什么](/img/ai-infra/sglang-parallelism/01-parallel-basics.png)

*图 1：TP 和 PP 让多张卡共同承载一个逻辑模型；普通 DP 复制逻辑模型，处理不同请求。示意图省略 embedding、输出头和通信细节。*

还有一个贯穿全文的判断：**降低单请求延迟、提高总吞吐、扩大上下文容量，是三个不同目标。** 某种策略能让服务接住更多请求，并不意味着一条请求会更快。

## 02｜TP：每层一起算，通信也跟着每层走

张量并行（Tensor Parallelism，TP）把某些权重矩阵按行或列切开，各 GPU 计算自己负责的部分，再通过集合通信组织后续计算。

以 Transformer 的 MLP 为例，可以把上投影的输出维度分给不同 GPU，分别执行中间计算，再在下投影之后汇总结果。Attention 也可以按 head 等维度划分，但具体切法受模型结构约束。

它的直接收益，是分摊权重和部分计算。代价则是通信被放进了模型层的执行路径：常见形式包括 All-Reduce，或者配合布局与融合使用 Reduce-Scatter、All-Gather。

对大批量 Prefill，矩阵乘法比较饱满，通信可能被较多计算摊薄。对小批量 Decode，每步新增 token 很少，单次计算变短，集合通信的固定延迟就更容易显露出来。

因此，**TP 越大，并不意味着延迟越低。** 从 4 卡增加到 8 卡，计算量虽然被进一步分散，却也可能让每卡矩阵变小、通信参与方变多。结果取决于模型、batch、内核和互联。

显存也不能简单套用“全部除以 TP”。权重、激活、通信缓冲和 KV Cache 的切分方式并不相同。特别是 GQA 的 KV head 数有限，而某些 MLA 路径在 Attention TP 各 rank 上保存同一份 latent KV，增加 TP 不会等比例降低这部分缓存。[2][3]

实际部署时，通常值得先在高速互联域内验证 TP，再考虑跨节点扩展；是否更快，最终仍要看相同负载下的延迟与吞吐。

## 03｜普通 DP：复制服务能力，也复制缓存边界

普通数据并行（Data Parallelism，DP）把不同请求分配给不同模型副本。在推理场景中，这些副本不需要像训练那样同步梯度。

这里的“副本”是一个完整的**逻辑模型实例**，不一定只占一张卡。一个副本内部仍然可以采用 TP 和 PP。

例如，8 张 GPU 可以组成两个 TP=4 的副本。请求 A 进入第一个副本，请求 B 进入第二个副本；两个请求不必参与同一组逐层计算。

这适合模型已经能装入较小卡组、主要压力来自并发的场景。但普通 DP 不会把一条超长请求自动拆给多个副本，也不会直接提高单副本的上下文容量。

它还引入了一个经常被低估的问题：**缓存属于哪个副本？**

多个副本通常各自维护运行状态与 KV／前缀缓存。把共享前缀的请求送到同一个副本，可能节省 Prefill；如果这个副本已经排队，又可能损失首 token 延迟。负载均衡与缓存局部性需要一起考虑。

SGLang 提供原生 DP 调度，也可以由 Model Gateway（SMG）路由到多个独立 worker。两者的进程管理、路由与扩缩容边界不同。本文对应版本的上游指南推荐生产级 DP 使用 SMG，并列出原生 DP 在缓存感知路由、容错与熔断等方面的限制。这里的建议针对部署实现，不改变“多副本分流”的 DP 原理。[3]

在每 GPU 一个计算 rank 的常见部署下，若有 R 个普通副本，每个副本使用 P 个流水线阶段、每阶段 T 张卡，则总卡数可由源码中的 `compute_world_size` 对应关系得到：[1]

```text
总 GPU 数 = R × P × T
```

这个公式描述的是独立副本。后面出现的 DP Attention，要按另一种方式理解。

## 04｜PP：按层切分，用流水线换取模型容量

流水线并行（Pipeline Parallelism，PP）把连续的模型层放到不同 stage。例如前半部分层放在第一组 GPU，后半部分层放在第二组 GPU，stage 之间传递激活。

每个 stage 内部仍可使用 TP。假设两台机器各有 8 张 GPU，可以令每台内部 TP=8，两台之间 PP=2，总共使用 16 张卡。节点数只是物理放置方式，不能再额外乘一次。

这种组合把较频繁的 TP 通信留在节点内，把跨 stage 的通信放到节点之间，是跨节点部署的一种候选方案。不过收益依赖各 stage 的执行时间与调度是否匹配，以及实际网络与负载。执行负载均衡不等于层数必须相同：上游针对 DeepSeek-V3.1 给出的调优经验是，层数不能整除时，把较大的分片放到更高的 PP rank；这属于特定场景的经验，不是通用最优划分。[4]

PP 的核心代价是**流水线气泡**。一个请求经过 stage 1 时，stage 2 可能还在等待；接近批次结束时，也会出现部分 stage 提前空闲。

在“各 stage 等时、忽略通信和调度开销”的简化前向模型里，P 个 stage 连续处理 M 个微批次，利用率可近似写为：

```text
利用率 ≈ M / (M + P − 1)
```

这个式子只解释为什么更多可重叠工作有利于填满流水线，不是 SGLang 在线服务的性能预测公式。

对单条自回归请求，后一个 token 依赖前一个 token 的结果。PP 不会消除这种依赖。SGLang 支持将 PP 与 chunked prefill 等机制结合，让长输入的多个分块形成流水，但分块大小、stage 数量和调度都会影响最终表现。[4]

对长输入，还可以关注 `--enable-dynamic-chunking`。即使每块 token 数相同，后面的块也可能因历史前缀变长而耗时增加。上游实现用运行时间模型调整后续块大小，以缓解分块耗时不均造成的流水线气泡。[4]

因此，PP 首先值得用于解决容量与跨节点组织问题；动态分块则提供了进一步调优的方向。是否改善首 token 延迟（TTFT）或 Decode 延迟，仍需针对工作负载验证。

## 05｜DP Attention：Attention 分请求，其他层继续协作

DP Attention（上游文档也缩写为 DPA）最容易被误解成“再复制几份完整模型”。实际上，它改变的是**Attention 这一部分计算的并行布局**。

以 MoE 推理为例，不同 Attention DP 组处理不同请求批次；进入 MoE 等其他计算阶段后，仍按相应的通信组组织 token 和计算。整个引擎并没有因此变成多个互不相关的完整模型副本。

这给了 Attention 与 MoE 分别选择布局的空间：Attention 可以少做一些重复 KV 存储与读取，MoE 则在更大的 rank 范围内组织专家计算。[3]

![普通 DP 和 DP Attention 在 8 张 GPU 上的区别](/img/ai-infra/sglang-parallelism/02-dp-vs-dpa.png)

*图 2：下半图的 Attention 与 MoE 是同一组 GPU 在不同层的布局，不是两套硬件。具体组合仍需模型和通信后端支持。*

假设一个 stage 使用 8 个 rank，开启 DP Attention，Attention DP=4，且 Attention CP=1，那么每个 Attention 组的 TP 宽度为 2。

在 MLA 的相应实现中，这有助于减少同一请求的 latent KV 在整个 8 卡范围内的重复；但一个 Attention TP 组内部仍可能存在复制。不同请求如果拥有相同前缀，也不会因此天然共享跨组缓存。

代价是层与层之间可能需要重新组织 token、汇集或分散数据，并处理不同 Attention DP 组的负载差异。DP Attention 应结合模型结构、KV 压力和 MoE 路径评估，不能套用“开了就更快”的结论。

一个很实用的识别方法是：**普通 DP 的边界是完整模型副本；DP Attention 的边界是 Attention 计算组。**

## 06｜EP：分配专家，也要选择通信方式

专家并行（Expert Parallelism，EP）用于 MoE 模型。模型拥有多组专家参数，路由器为每个 token 选择若干专家；EP 把这些专家放到不同 GPU 上。

在 All-to-All 类后端中，一次 MoE 计算可以概括为：路由器选专家，dispatch 把 token 的表示送到专家所在 GPU，各卡执行专家计算，再通过 combine 把结果送回并合并。

还要区分 `--moe-a2a-backend none` 路径。上游文档将它列为默认选项：各 EP rank 对同一批 token 计算自己负责的部分，再合并结果，通信可能涉及 All-Reduce／All-Gather。这不等于没有数据搬运，只是数据组织与通信成本不同于按专家定向的 All-to-All dispatch/combine。[5][6]

与 TP 切分一个权重矩阵相比，EP 更接近“把不同专家放在不同位置”。同一个专家内部是否继续切分，则是 MoE Tensor Parallelism 的问题。

EP 的关键成本包括三部分：

- **通信成本。** token 可能需要跨卡或跨节点移动，常见路径涉及 All-to-All 类操作。
- **负载倾斜。** 某些专家被选得更多，持有这些专家的 GPU 可能成为慢点。
- **计算粒度。** 每个专家实际收到的 token 太少，矩阵计算未必高效。

因此，EP 规模不能只依据专家总数来选，还要看每步的 token 数、路由分布，以及通信域有多大。

SGLang 的 EPLB 用于调整专家的物理放置或冗余副本，以缓解负载不均。应区分模型选择的**逻辑专家**与 GPU 上承载它的**物理副本**；负载均衡不等于任意改变模型的路由语义。[5]

EP 也不能简单乘在 TP 之上。在当前实现中，MoE 的 EP、MoE TP 和 MoE DP 是同一 rank 域上的不同组织方式。不同 All-to-All 后端对 EP／TP 组合还有额外限制，参数算得整除，只是配置成立的第一步。[1][5]

例如，本文版本的官方文档明确列出：DeepEP、Mooncake、NIXL-EP、MORI、`pplx`、`ascend_fuseep` 要求 `ep_size = tp_size`；文档将 EP 小于基础 TP 的混合布局列为 `none` 后端的支持场景。其他后端和后续版本仍应分别核对，不能把一条限制套到所有实现。[5]

## 07｜CP 与 DCP：让多张卡处理同一条上下文

**Prefill CP 关注长输入的协同计算。**

上下文并行（Context Parallelism，CP）沿序列维度分配 token，让多个 rank 共同处理一条长输入，并通过通信获得完成 Attention 所需的信息。

因果 Attention 中，靠后的 token 能看到更长的历史。简单地把前半段、后半段各放一张卡，可能造成计算不均衡。因此 SGLang 提供的相关路径包含 zigzag、interleave 等策略，但支持情况依模型而异。[6]

这里必须区分 CP 与 chunked prefill：后者把 Prefill 切成调度分块，便于安排执行与资源；前者定义多个 rank 如何共同承担上下文。两者可以组合，却不是同一个开关。

**DCP 关注 Decode 时的历史 KV。**

生成一个新 token 时，Attention 需要访问历史上下文。对支持的 MLA 路径，DCP 将同一请求的目标模型 KV 按 token 位置条带化存储到多个 rank，让各 rank 读取自己的分片，再合并局部 Attention 结果。[2]

![DCP 按 token 位置切分受支持的目标模型 MLA KV](/img/ai-infra/sglang-parallelism/03-dcp.png)

*图 3：受支持目标模型 MLA KV 的 4 路 DCP 概念示意，token 位置从 0 开始。局部 Attention 输出需要结合各自的归一化信息合并，不能直接求平均。*

不同分片的 Attention 分数总量不同，直接平均会丢失相应的权重。正确合并需要保留各分片的 softmax 归一化信息，例如 log-sum-exp（LSE），据此加权局部输出。

这是一笔明确的交换：减少随上下文增长的 KV 存储和读取，增加局部结果交换与归并。短上下文未必能摊薄新增通信；而 DCP 节省的是受支持路径中的 KV，不代表全部模型状态、draft KV 都按同样比例下降。

还要牢记 DPA 与 DCP 的区别：**DPA 将不同请求分组；DCP 将同一请求的历史上下文分片。**

在本文核对的组合中，DCP 组需要包含在一个 Attention TP 组内。比如 TP=8、Attention DP=4、Attention CP=1 时，Attention TP=2，可以讨论组内 DCP=2；DCP=4 会跨越不同请求组，属于无效拓扑。[2]

**基础 TP 能被 DCP 整除，还不足以保证这个组合正确。** 本文版本的 DCP 文档明确提醒：启动校验未覆盖更强的 Attention TP 嵌套条件；已核对的通信组初始化代码也只检查基础 TP 的整除。因此部署者需要额外确认 `attn_tp_size % dcp_size == 0` 及组的包含关系，不能仅凭某处检查通过就判断拓扑有效。[1][2]

## 08｜PD 分离：拆开的是服务阶段

Prefill 与 Decode 的工作特征不同：Prefill 一次处理较多输入 token；Decode 逐步生成，持续访问历史 KV。两类工作共享资源时，可能互相影响延迟与调度。

Prefill/Decode Disaggregation（PD 分离）把它们安排到不同 worker 或资源池：Prefill worker 处理输入并产生所需 KV，Decode worker 接收相应状态，继续生成。[7]

![流水线并行与 PD 分离的区别](/img/ai-infra/sglang-parallelism/04-pp-vs-pd.png)

*图 4：PP 按层划分同一次模型执行；PD 按服务阶段划分资源。图中省略路由、首 token 返回时机与具体传输实现。*

这与 PP 有本质区别。PP 中，一个阶段负责部分模型层；PD 中，Prefill 池和 Decode 池各自具备执行相应阶段完整模型计算的能力，其内部还可以使用 TP、PP、EP 等策略。

PD 的收益来自分别安排两类工作、独立配置容量和减少阶段间干扰。成本则包括 KV 传输、连接与协调、两边排队，以及传输失败时的恢复处理。

如果 Prefill 后要搬运的 KV 很大，而网络又不够快，隔离计算带来的收益可能被传输开销抵消。两边使用不同并行配置时，还必须确认传输引擎支持对应的 KV 布局转换，不能假定所有 TP／CP／DCP 组合都能互通。

PD 的总资源，应把两个池分别计算后相加，而不是相乘。多模态场景还可以进一步做 Encoder/Prefill/Decode 分离：Encoder 处理图像等输入，后续阶段消费编码结果；这又增加了一种阶段边界。[8]

## 09｜最容易算错的，是这些并行度的组合

阅读 SGLang 参数时，尤其要留意 `tp_size` 的上下文。在混合布局下，它还承担了一个 stage 内基础 rank 域大小的含义，Attention 或 MoE 真正使用的 TP 宽度可能更小。

当前源码的宽度推导，可以用下面两组关系理解。式中的 DP、TP 分别指所在计算侧的宽度，不能互换：[1]

```text
Attention 侧：T = DP × CP × TP
MoE 侧：T = DP × EP × TP
```

这里的 T 是同一个 stage 的基础 rank 域；普通 DP 的独立副本位于这组关系之外。未启用的维度按 1 处理。DCP 则在受支持布局的 Attention TP 组内组织，不是又增加一批 GPU。

另一个容易误配的地方是 **`--dp-size` 的双重语义**：未开启 `--enable-dp-attention` 时，它表示普通 DP 副本数，会乘进总卡数；开启后，它表示 Attention DP 宽度，不再额外增加一批 GPU。普通 DPA 参数路径要求 `tp_size % dp_size == 0`；上游指南还说明，`dp_size` 为 1 时 DPA 会关闭。[1][3][6]

三个 8 卡例子，可以帮助检查自己的理解：

**方案 A：普通 DP=2，每个副本 TP=4。** 两个完整逻辑模型实例，总共 8 张卡。适合讨论并发分流。

**方案 B：基础 TP=8，Attention DP=4，EP=8。** 在 Attention CP=1、MoE DP=1 时，Attention TP=2、MoE TP=1。这些布局复用同一组 8 张卡，不能算成 8×4×8。

方案 B 对应下面的并行参数片段。它仅表达布局，省略模型与后端等参数，未作为完整命令执行：

```text
--tp-size 8
--ep-size 8
--dp-size 4
--enable-dp-attention
```

**方案 C：方案 B 再讨论 DCP=2。** DCP 嵌套在每个两卡 Attention TP 组内，总卡数仍为 8。是否能启用，还要核对模型、Attention 后端、KV 管理方式和部署阶段。

这些是拓扑计算示例，不是通用推荐配置。实际校验还包括 head 数与权重维度的整除、通信后端约束、量化实现，以及功能之间的兼容性。

此外，还有几类名字相近但用途不同的机制：LayerNorm sequence parallelism 针对受支持 TP 路径中的归一化与残差计算布局；MoE DP 为 MoE 引入副本维度；多模态 Encoder DP 分配编码任务。它们都有特定实现范围，不能当成所有模型通用的额外并行开关。[6]

同一快照中的 DWDP 通过权重预取组织特定 MoE Prefill 路径，Elastic EP 则涉及运行时的专家并行规模调整。二者都带有额外实现与兼容约束，也不应直接当作 GPU 数量公式的新乘数。[6]

计算与通信重叠、CUDA Graph、投机解码、前缀缓存同样重要，但它们分别作用于执行、解码或缓存，不应直接作为 GPU 数量公式中的新维度。

## 10｜选型时，从瓶颈出发，再决定怎么拆

**第一步，确定约束。** 是权重装不下、KV 装不下，还是延迟不达标？同时确认 GPU 之间是 NVLink／NVSwitch、PCIe，还是跨节点网络。

**第二步，找到较小且可用的单副本配置。** 模型容量问题优先考察 TP／PP，MoE 再加入 EP。先留出 KV 和执行缓冲空间，再讨论把卡数压到多低。

**第三步，按工作负载选择扩展方向。** 短请求、高并发可以比较增加副本与扩大 TP；长输入关注 Prefill、CP 与调度；长上下文生成关注 KV 布局和 DCP；MoE 关注专家负载与通信；阶段干扰明显时，再评估 PD 分离。

**第四步，用同一组服务指标比较。** 固定模型、精度、输入输出长度分布、到达率或并发，以及缓存命中条件，至少观察：

- TTFT：从请求发出到首 token 的时间，包含排队等服务开销。
- TPOT（time per output token）：生成阶段的平均每输出 token 时间；ITL（inter-token latency）：相邻 token 的时间间隔。明确统计口径，并关注尾部延迟。
- 在延迟目标以内完成的请求数、输出 token 吞吐，以及单位 GPU 的有效产出。
- KV 占用、缓存命中、通信耗时；MoE 场景还要看专家负载分布。

一个配置即使总吞吐更高，如果大量请求超过延迟目标，也未必是更好的线上方案。反过来，一个能容纳更长上下文的配置，即便短请求变慢，也可能更符合业务需求。

SGLang 提供了多种组织 GPU 的方式。真正需要回答的，是每次切分之后，**哪些计算变小了，哪些数据需要移动，哪些状态仍然重复，以及这些变化是否解决了当前瓶颈。**

## 参考资料

以下源码链接固定到本文核对的提交；文档导航页会继续更新。

1. SGLang 并行宽度推导、通信组构造与卡数计算：[runtime_context.py](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/python/sglang/srt/runtime_context.py#L138)、[parallel_state.py](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/python/sglang/srt/distributed/parallel_state.py#L2507)、[server_args.py 的 compute_world_size](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/python/sglang/srt/server_args.py#L581)。
2. SGLang：[Decode Context Parallelism](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/docs/docs/advanced_features/dcp.mdx)。
3. SGLang：[DP、DP Attention 与 Model Gateway 指南](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/docs/docs/advanced_features/dp_dpa_smg_guide.mdx)、[DP Attention 实现](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/python/sglang/srt/layers/dp_attention.py)。
4. SGLang：[Pipeline Parallelism](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/docs/docs/advanced_features/pipeline_parallelism.mdx)。
5. SGLang：[Expert Parallelism](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/docs/docs/advanced_features/expert_parallelism.mdx)。
6. SGLang 并行参数与兼容约束：[parallel.py](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/python/sglang/srt/arg_groups/fields/parallel.py)、[parallel_hook.py](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/python/sglang/srt/arg_groups/parallel_hook.py)。
7. SGLang：[Prefill/Decode Disaggregation](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/docs/docs/advanced_features/pd_disaggregation.mdx)。
8. SGLang：[Encoder/Prefill/Decode Disaggregation](https://github.com/sgl-project/sglang/blob/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251/docs/docs/advanced_features/epd_disaggregation.mdx)。
