---
title: SGLang 并行策略分析：一组 GPU，究竟该怎么拆？
description: 基于 SGLang v0.5.19，从切分对象、通信代价和显存布局出发，分析 SGLang 的 TP、DP、PP、EP、CP/DCP 与 PD 分离，并用 8 卡示例解释组合关系和选型边界。
date: 2026-09-18
---

# SGLang 并行策略分析：一组 GPU，究竟该怎么拆？

同样是 8 张 GPU，为什么有时用 TP=8，有时拆成两个 TP=4 的副本，而部署 MoE 模型时，又会同时配置 TP、DP Attention 和 EP？

先设定一个便于比较的场景：手里有 8 张卡，模型在 TP=4 时能够运行，也能留出足够的 KV Cache 和执行缓冲空间。现在请求越来越多，需要决定这 8 张卡怎么用。

一种做法是让 8 张卡一起算每个请求；另一种是分成两组，各处理一批请求。前者分摊单个模型的计算，后者增加完整模型副本。两种方式消耗相同的卡数，却改变了不同的东西。

下面从这两个配置开始，逐步讨论模型容量、MoE、长上下文和 Prefill/Decode 干扰。**每换一种布局，都跟着请求看一遍：计算落在哪些卡上，数据要在哪里移动。**

> 本文基于 SGLang 正式版 [v0.5.19](https://github.com/sgl-project/sglang/releases/tag/v0.5.19)，源码与文档固定到该标签对应的提交 `0bcd822377da`，查阅日期为 2026 年 9 月 18 日。8 卡例子用于说明布局，没有做 GPU 性能实测；切换模型或并行策略时，还需满足对应的容量、模型与后端约束。

## 01｜先比较：一个 TP=8，还是两个 TP=4？

**先把 8 张卡组织成一个 TP=8 实例。**

张量并行（Tensor Parallelism，TP）把模型中的部分权重矩阵按行或列切分。每张卡计算自己负责的部分，再通过集合通信组织后续计算。

以 Transformer 的 MLP 为例，上投影的输出维度可以分给不同 GPU，各卡完成中间计算，再在下投影之后汇总结果。Attention 也可以按 head 等维度划分，具体切法受模型结构约束。

因此，一个请求进入 TP=8 实例后，会由这 8 张卡共同完成模型计算。它通常和其他请求组成 batch 执行，并不意味着整个实例一次只能服务一个请求。

从 TP=4 扩大到 TP=8，每张卡承担的权重和部分计算减少了，但参与集合通信的卡更多了。常见通信包括 All-Reduce，以及配合布局与融合使用的 Reduce-Scatter、All-Gather。

对大批量 Prefill，矩阵计算量较大，通信开销可能被计算摊薄；对小批量 Decode，每步新增 token 很少，单次计算较短，通信的固定延迟就更明显。**把 TP 翻倍，不能直接推导出延迟减半。**

**再把同样的 8 张卡拆成两个 TP=4 实例。**

这就是普通数据并行（Data Parallelism，DP）：每个实例拥有完整的逻辑模型，不同请求交给不同实例。这里一个副本占 4 张卡，副本内部仍然使用 TP。

请求 A 进入 GPU 0–3，请求 B 进入 GPU 4–7。两组各自调度、各自执行模型，不需要为这两个请求参与同一组逐层计算，也不需要像训练那样同步梯度。

![TP、PP、DP 分别切分什么](/img/ai-infra/sglang-parallelism/01-parallel-basics.png)

*图 1：TP 按张量切分，PP 按模型层切分，普通 DP 复制完整逻辑模型。图中省略 embedding、输出头和通信细节；PP 在下一节展开。*

两个副本多了分流请求的选择，也多了缓存边界。假设请求 C 与 A 共享很长的前缀，把 C 送到第一组，可能复用已有缓存；但如果第一组已经排队，等待时间又可能抵消省下的 Prefill。路由器需要同时考虑缓存位置和实时负载。

SGLang 提供原生 DP 调度，也可以由 Model Gateway（SMG）路由到多个独立 worker。《DP、DP Attention 与 Model Gateway 指南》推荐生产级 DP 部署使用 SMG，并列出了原生 DP 在缓存感知路由、容错与熔断等方面的限制。[3]

总卡数可以直接看 [`server_args.py` 的 `compute_world_size`](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/server_args.py#L4173-L4182)。下面节选其返回语句，仅调整换行：

```python
return (
    (1 if enable_dp_attention else dp_size)
    * tp_size
    * pp_size
)
```

未开启 DP Attention 时，第一项取 `dp_size`。我们的两个 TP=4 副本使用 `2 × 4 × 1 = 8` 张卡；单个 TP=8 实例则是 `1 × 8 × 1 = 8` 张卡。

到这里，两种方案的比较条件已经清楚：模型能在 4 卡上运行时，可以比较扩大 TP 与增加副本。普通 DP 让不同请求有机会分开执行，却不会把一条超长请求自动拆给两个副本。如果连一个 TP=4 副本都装不下，就要先解决模型容量问题。

## 02｜模型需要更多卡时，PP 把通信移到层与层之间

现在改变一个条件：仍然只有 8 张卡，但它们位于两台机器，每台 4 张；模型无法在单台 4 卡上承载。

一种选择是跨节点 TP=8。另一种选择是每台内部 TP=4，两台之间 PP=2：GPU 0–3 负责一部分连续模型层，GPU 4–7 负责后续层。这就是流水线并行（Pipeline Parallelism，PP）。总卡数仍为 `TP × PP = 4 × 2 = 8`。

跟着一个请求看，它先在第一组完成前面几层，把激活传给第二组，再完成后面的层。较频繁的 TP 通信留在节点内，两个 PP stage 之间通过跨节点通信传递激活。是否优于跨节点 TP，要看各 stage 的耗时、网络和调度。

这种安排增加了一个等待问题：第一组正在处理某个微批次时，第二组可能还没拿到输入；处理接近结束时，又可能有 stage 提前空闲。这些空闲就是**流水线气泡**。

在各 stage 耗时相同、忽略通信与调度开销的简化前向模型中，P 个 stage 处理 M 个微批次，利用率近似为：

```text
利用率 ≈ M / (M + P − 1)
```

代入两阶段流水线，只有一个微批次时约为 `1/2`；连续处理四个微批次时约为 `4/5`。这个推导解释了为什么增加可重叠工作能减少空闲，并不能拿来预测 SGLang 在线服务的实际利用率。

长输入可以借助 chunked prefill 形成更多分块，供流水线安排。但相同 token 数的块不一定耗时相同：越靠后的块，通常需要看到越长的历史前缀。`--enable-dynamic-chunking` 用运行时间模型调整后续块大小，缓解分块耗时不均造成的气泡。[4]

stage 的负载也不能只按层数判断。《Pipeline Parallelism for Long Context》针对 DeepSeek-V3.1 给出的经验是：层数不能整除时，把较大的分片放到更高的 PP rank。换模型后仍要测量各 stage 的执行时间。[4]

PP 解决了模型如何跨卡组承载的问题，但单条自回归请求的后一个 token 仍依赖前一个 token。它没有消除生成过程的串行依赖，也不会自动缩短每个 token 的生成时间。

前两节都把模型作为一个整体拆分。换到 MoE 模型，还会遇到另一种矛盾：Attention 与专家层未必适合用相同的并行布局。

## 03｜MoE 的第一处分歧：Attention 不一定需要 8 卡一起算

下面回到一个 stage 使用 8 张卡的场景，选择支持 DPA 的 MoE 模型。先不启用 PP，也不启用 Attention CP。

对部分 absorbed MLA 路径，Attention TP 各 rank 保存并读取同一请求的 latent KV。TP 能切分相关权重和 head，却不会把这份缓存也等比例切开。于是，8 卡一起执行 Attention 时，同一请求的 KV 可能在 8 个 rank 上重复保存。[2][3]

如果现在主要压力来自 KV，而专家层又需要较大的卡组，直接复制两个完整模型未必合适。DP Attention（DPA）允许只对 Attention 按请求分组，其他层继续采用相应的并行布局。

**仍然用这 8 张卡，把 Attention DP 设为 4。** Attention CP=1 时，每组 Attention TP=2：请求 A 所在批次交给一组两张卡，请求 B 所在批次可以交给另一组。进入 MoE 层后，再按专家布局组织 token 和计算。

![普通 DP 和 DP Attention 在 8 张 GPU 上的区别](/img/ai-infra/sglang-parallelism/02-dp-vs-dpa.png)

*图 2：普通 DP 的两组各自承载完整逻辑模型。DPA 图中的 Attention 与 MoE 则复用同一组 GPU，只是在不同层采用不同布局。*

这组数字来自 [`runtime_context.py` 的 `derive_attention_widths`](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/runtime_context.py#L134-L144)（节选，仅调整换行）：

```python
attn_dp_size = dp_size if enable_dp_attention else 1
return (
    attn_dp_size,
    tp_size // attn_dp_size // attn_cp_size,
)
```

开启 DPA 后，`attn_dp_size` 取 `dp_size`，Attention TP 则是 `8 // 4 // 1 = 2`。再看第 01 节的 `compute_world_size`：开启 DPA 时，卡数公式的第一项变成了 1。因此，**这里的 DP=4 是 Attention 的分组宽度，没有额外增加 4 份完整模型。**

对上述 MLA 路径，请求 A 的 latent KV 不必再铺满整个 8 卡范围，但所在的两卡 Attention TP 组内仍可能有复制。不同组的请求即使前缀相同，也不会因此天然共享跨组缓存。

DPA 减少了这类重复，也增加了层间数据重组的工作：从分组 Attention 进入 MoE 时，token 可能需要汇集或分发，各 Attention 组的负载差异也需要处理。是否有收益，要把节省的 KV 存储、读取和新增通信一起计算。

接下来沿着请求 A 继续走：它离开两卡 Attention 组后，进入的专家层究竟由哪些 GPU 计算？

## 04｜进入专家层：EP 决定专家放在哪里

MoE 的路由器为每个 token 选择若干专家。专家并行（Expert Parallelism，EP）把这些专家分配到不同 GPU；专家内部还可以继续做张量并行，即 MoE TP。

仍用上一节的 8 张卡，设基础 TP=8、EP=8、MoE DP=1。MoE TP 的计算位于 [`derive_parallel_widths`](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/runtime_context.py#L147-L182) 的返回字典中（节选，仅调整换行）：

```python
"moe_tp_size": (
    tp_size // moe_ep_size // moe_dp_size
),
```

代入后是 `8 // 8 // 1 = 1`。这表示此布局中专家没有再沿 MoE TP 拆开，而是通过 EP 分布到这 8 个 rank 上。前一节的 Attention DP=4 和这里的 EP=8，使用的仍是同一组卡。

如果选择 All-to-All 类后端，请求 A 的 token 会先经过专家路由；dispatch 将 token 表示送到目标专家所在 GPU；各卡执行专家计算后，再由 combine 送回并合并结果。读到 EP 参数时，需要同时看这条通信路径。

`--moe-a2a-backend none` 则采用不同的组织方式。《Expert Parallelism》文档将其列为默认选项：各 EP rank 对同一批 token 计算自己负责的部分，再汇总结果，通信可能涉及 All-Reduce／All-Gather。`none` 并不代表没有数据搬运。[5][6]

专家分开以后，每张卡承担的工作也不一定均匀。假设一批 token 集中选择少数专家，持有这些专家的 GPU 就可能拖慢这一步；如果每个专家收到的 token 太少，矩阵计算又可能过于零碎。EP 的规模需要结合 token 数、路由分布和互联条件来选。

SGLang 的 EPLB 通过调整专家物理放置或冗余副本缓解负载不均。模型选中的逻辑专家与 GPU 上承载它的物理副本需要分开理解，负载均衡不等于任意改变路由语义。[5]

布局的算术关系还不是完整兼容条件。《Expert Parallelism》列出的 DeepEP、Mooncake、NIXL-EP、MORI、`pplx`、`ascend_fuseep` 要求 `ep_size = tp_size`；EP 小于基础 TP 的混合布局属于 `none` 后端支持的场景。选择其他后端时，要检查其各自约束。[5]

现在我们已经为 Attention 和 MoE 分别选了布局。但如果请求 A 本身越来越长，把其他请求分出去也无法消除 A 的上下文开销。

## 05｜一条请求太长：CP 和 DCP 拆分上下文

前面的 DPA 把不同请求交给不同 Attention 组。上下文并行（Context Parallelism，CP）则让多个 rank 共同处理同一条请求的上下文。

在 Prefill 阶段，一条长输入包含许多 token。CP 沿序列维度分配工作，各 rank 通过通信取得完成 Attention 所需的信息。因果 Attention 中，后面的 token 能看到更多历史；如果简单按前后两半分配，计算可能不均衡。SGLang 的相关路径提供 zigzag、interleave 等策略，具体支持情况取决于模型。[6]

这与 PP 一节里的 chunked prefill 是两件事：chunked prefill 把输入切成便于调度的分块，CP 决定多个 rank 如何共同承担上下文。切成多个调度块，并不自动意味着这些块在多张卡上并行执行。

**到了 Decode 阶段，压力转向了已有的历史 KV。**

继续使用 Attention DP=4、Attention TP=2 的 8 卡布局。在支持的 MLA 路径中，同一请求的 KV 仍可能在两卡 Attention 组内复制。Decode Context Parallelism（DCP）可以进一步按 token 位置拆开这份 KV，让两个 rank 各保存、读取自己的分片，再合并局部 Attention 结果。[2]

用 token 位置从 0 开始的示意来说，DCP=2 时，一张卡保存位置 0、2、4……，另一张卡保存 1、3、5……。下面用四路分片把这种分配方式画得更清楚：

![DCP 按 token 位置切分受支持的目标模型 MLA KV](/img/ai-infra/sglang-parallelism/03-dcp.png)

*图 3：四路 DCP 的局部示意，需要至少四个 rank 的 Attention TP 组。若在 8 卡例子中采用此图布局，可以讨论 Attention DP=2、Attention TP=4、DCP=4；它不是前述 DPA=4 的两卡组布局。*

各 rank 只看到一部分上下文，输出也只是局部结果。合并时不能直接求平均，需要结合各分片的 softmax 归一化信息，例如 log-sum-exp（LSE），加权得到完整结果。[2]

v0.5.19 的 `--dcp-comm-backend` 通用默认值是 `ag_rs`，也提供 `a2a`、`fi_a2a`。Kimi K3 有专门的默认覆盖，会按平台选择通信后端；分析 DCP 的通信成本时，需要查看具体模型最终使用的路径。[2]

这会减少每卡的 KV 存储和读取，但增加局部结果的交换与归并。短上下文可能不足以摊薄新增通信；DCP 节省的也只是受支持路径中的 KV，其他模型状态和 draft KV 不一定按相同比例下降。

**DCP 组必须位于同一个 Attention TP 组内。** 回到 Attention DP=4、Attention TP=2 的例子，DCP=2 可以放在组内；DCP=4 则会跨越处理不同请求的 Attention 组，拓扑无效。

这个约束为何容易漏掉，可以看 [`parallel_state.py` 的分组代码](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/distributed/parallel_state.py#L2493-L2499)（节选，仅调整换行）：

```python
if decode_context_parallel_size > 1:
    dcp_group_ranks = []
    for tp_group in group_ranks:
        for start in range(
            0, len(tp_group),
            decode_context_parallel_size,
        ):
            dcp_group_ranks.append(
                tp_group[
                    start : start + decode_context_parallel_size
                ]
            )
```

循环从基础 TP 组取出 rank，每次截取 `decode_context_parallel_size` 个，并没有在这里重新按 Attention DP 划界。因此，“8 能被 4 整除”并不能证明上述 DCP=4 配置正确。

《Decode Context Parallelism》文档也提醒，启动校验未覆盖更强的 Attention TP 嵌套条件。部署时除了检查 `attn_tp_size % dcp_size == 0`，还要确认通信组的实际包含关系。[1][2]

至此，我们讨论的都是模型内部如何组织计算。如果长 Prefill 和正在生成的 Decode 相互干扰，还可以在服务层把两个阶段拆开。

## 06｜PD 分离：给 Prefill 和 Decode 各自安排资源

Prefill 一次处理较多输入 token；Decode 持续逐步生成，并访问历史 KV。两类工作共用资源时，长 Prefill 可能影响正在生成请求的延迟，而大量 Decode 也会占用新请求所需的资源。

Prefill/Decode Disaggregation（PD 分离）分别设置 Prefill worker 和 Decode worker。请求 A 先在 Prefill 侧处理输入、产生所需 KV，再将相应状态传给 Decode 侧，继续生成。[7]

为了与前面比较，重新采用“完整模型可在 4 卡上运行”的容量假设：8 张卡可以分成一个 TP=4 的 Prefill worker 和一个 TP=4 的 Decode worker。卡数仍是 8，但它们承担的工作已经不同于普通 DP 的两个副本。

![流水线并行与 PD 分离的区别](/img/ai-infra/sglang-parallelism/04-pp-vs-pd.png)

*图 4：PP 按层划分同一次模型执行；PD 按服务阶段划分资源。图中省略首 token 返回时机、路由和具体传输实现。*

在 PP=2 的例子中，请求每次完成模型前向计算，都要依次经过两组负责不同层的 GPU。PD 则是 Prefill 池和 Decode 池各自具备执行相应阶段完整模型计算的能力；转交 KV 后，后续 Decode 在 Decode 池内继续。每个池内部还可以采用 TP、PP、EP 等布局。

拆开之后，可以分别配置两个阶段的容量，减少阶段间资源干扰。新增的成本包括 KV 传输、连接与协调、两边的排队，以及传输失败后的恢复处理。

例如，请求 A 的输入很长，产生的 KV 很大。如果传输时间过长，Prefill 侧省下的排队时间可能又花在了等待 KV 上。PD 的效果需要看完整请求路径，不能只比较两个 worker 各自的计算速度。

两边使用不同 TP／CP／DCP 配置时，还要确认传输引擎支持对应的 KV 布局转换。前面的四卡加四卡只是资源记账示例，不是任意模型、后端都能运行的命令。

多模态服务还可以进一步拆出 Encoder，形成 Encoder/Prefill/Decode 分离：Encoder 处理图像等输入，后续阶段消费编码结果。资源仍然按各个独立池分别计算后相加。[8]

## 07｜把这些配置放到一张 GPU 账单上

前面的例子反复使用 8 张卡，但参数含义不同。最容易算错的是把所有带 size 的参数都乘起来。

**普通 DP、TP、PP 决定独立副本的基本卡数。** 在每 GPU 一个计算 rank、未开启 DPA 的常见部署下：

```text
总 GPU 数 = 普通 DP × PP × TP
```

**DPA、MoE EP 等参数则可能重新划分同一个 stage 的 rank。** 令 T 为该 stage 的基础 rank 数，由前面的两处宽度推导可得：

```text
Attention：T = Attention DP × CP × Attention TP
MoE：      T = MoE DP × EP × MoE TP
```

未启用的维度按 1 处理。两行是在同一批卡上对不同计算进行分组，不应彼此相乘；DCP 再嵌套在受支持的 Attention TP 组内，也不另加一批卡。[1]

因此，基础 TP=8、Attention DP=4、EP=8，在 Attention CP=1、MoE DP=1 时，表示 Attention TP=2、MoE TP=1，总共仍是 8 张卡。对应的并行参数片段如下，省略了模型和后端等设置，未作为完整命令执行：

```text
--tp-size 8
--ep-size 8
--dp-size 4
--enable-dp-attention
```

这里尤其要看 `--enable-dp-attention`：不开启时，`--dp-size` 表示普通 DP 副本数；开启后，它表示 Attention DP 宽度。普通 DPA 参数路径要求 `tp_size % dp_size == 0`，《DP、DP Attention 与 Model Gateway 指南》还说明，`dp_size=1` 时 DPA 会关闭。[1][3][6]

PD 则按资源池记账：先算 Prefill 池的 GPU 数，再算 Decode 池，两者相加。有多个独立 worker 时，各自的资源也要计入。

还有一些容易混在一起的名字：v0.5.19 的 LayerNorm sequence parallelism 用于 Qwen3 dense 的 Prefill，在纯 TP 路径中将归一化与残差计算放到按序列切分的激活上，要求 TP>1 及 NVLink／NVSwitch；MoE DP 为专家计算引入副本维度；Encoder DP 分配编码任务。DWDP 通过权重预取组织特定 MoE Prefill 路径，Elastic EP 涉及运行时专家并行规模调整。它们都有各自的适用范围，不宜当成通用卡数公式的新乘数。[6]

计算与通信重叠、CUDA Graph、投机解码和前缀缓存会影响性能，但应放回对应的执行或缓存路径分析，不能与 GPU 分组维度混算。

## 08｜回到部署选择：下一轮应该比较什么？

回到最初的模型：它能在 TP=4 时运行，并留下足够的缓存与缓冲空间。第一轮就可以比较一个 TP=8 和两个 TP=4 副本，固定模型、精度、输入输出长度分布、到达率或并发，以及缓存命中条件。

如果两个副本的吞吐更高，再检查共享前缀是否被分散、排队是否倾斜。如果 TP=8 的延迟更低，也要看这个收益能否覆盖实际业务的 batch 和请求长度，不能只凭一组短请求下结论。

若模型需要跨节点承载，再比较跨节点 TP 与节点内 TP 加 PP，重点看逐层通信、stage 耗时和流水线空闲。对于支持相应路径的 MoE 模型，则分别观察 Attention 的 KV 压力和 MoE 的专家负载，再选择 DPA、DCP、EP；有明显阶段干扰时，才进一步比较 PD 的完整请求路径。

这些比较至少需要同一组服务指标：

- **TTFT**：从请求发出到首 token，包含排队等服务开销。
- **TPOT 与 ITL**：前者是生成阶段的平均每输出 token 时间，后者是相邻 token 的时间间隔；保留统计口径并关注尾延迟。
- **延迟目标内的有效吞吐**：完成多少请求、生成多少输出 token，以及每张 GPU 的有效产出。
- **解释差异的指标**：KV 占用、缓存命中、通信耗时、stage 空闲时间；MoE 还要看专家负载分布。

总吞吐上升但更多请求超时，未必值得上线；短请求变慢但能够承载必要的长上下文，也可能符合业务目标。选定目标后，先找到浪费发生在哪一段，再选择改变那一段计算、存储或通信的并行策略。

## 参考资料

以下文档与源码链接均固定到 v0.5.19 对应的提交 `0bcd822377da`。

1. SGLang 并行宽度推导、通信组构造与卡数计算：[runtime_context.py](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/runtime_context.py#L134)、[parallel_state.py](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/distributed/parallel_state.py#L2342)、[server_args.py 的 compute_world_size](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/server_args.py#L4173)。
2. SGLang：[Decode Context Parallelism](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/docs/advanced_features/dcp.mdx)。
3. SGLang：[DP、DP Attention 与 Model Gateway 指南](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/docs/advanced_features/dp_dpa_smg_guide.mdx)、[DP Attention 实现](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/layers/dp_attention.py)。
4. SGLang：[Pipeline Parallelism](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/docs/advanced_features/pipeline_parallelism.mdx)。
5. SGLang：[Expert Parallelism](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/docs/advanced_features/expert_parallelism.mdx)。
6. SGLang 并行参数与兼容约束：[server_args.py 的并行参数](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/server_args.py#L937-L1140)、[parallel_hook.py](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/arg_groups/parallel_hook.py)。
7. SGLang：[Prefill/Decode Disaggregation](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/docs/advanced_features/pd_disaggregation.mdx)。
8. SGLang：[Encoder/Prefill/Decode Disaggregation](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/docs/advanced_features/epd_disaggregation.mdx)。
