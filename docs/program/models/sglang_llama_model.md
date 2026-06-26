# sglang llama 模型实现分析

> 首次编写：2026-06-26 | 最后更新：2026-06-26

## Llama架构中，张量（Tensor）在模型各层之间的流动过程。

``` shell
用户输入的自然语言文本 → Tokenize（分词器转换为 token ids）→ input_ids（模型输入的整数序列）
    ↓
Embedding（嵌入层查表获取向量）→ hidden_states（初始 token 语义表示）
    ↓
[Decoder Layer 0~N]（Transformer 解码器层逐层处理）→ hidden_states（不断丰富的上下文语义表示）
    ↓
RMSNorm（最终层归一化）→ final hidden_states（归一化后的最终隐藏状态）
    ↓
LM Head（语言模型头映射到词表）→ logits（词表大小的原始预测分数）
    ↓
LogitsProcessor（logits 处理器）→ LogitsProcessorOutput（包含 logits、logprobs 等的结构化输出）
    ↓
Sampler（采样器根据概率分布采样）→ 输出 token（选中的下一个 token id）
    ↓
Detokenize（分词器反向转换）→ 文本输出（最终生成的自然语言文本）
```

更详细:

``` shell
┌─────────────────────────────────────────────────────────────┐
│                    用户输入的自然语言文本                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Tokenize（分词器将文本转换为 token ids）                      │
│  输出: input_ids (shape: [batch_size, seq_len])              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Embedding（嵌入层查表获取向量）                               │
│  输出: hidden_states (shape: [num_tokens, hidden_size])      │
│  说明: 每个 token 映射为 hidden_size 维的稠密向量              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  [Decoder Layer 0~N]（Transformer 解码器层逐层处理）          │
│  每层包含:                                                    │
│    - Self-Attention（自注意力机制，捕获上下文依赖）            │
│    - MLP（前馈神经网络，非线性变换）                           │
│  输出: hidden_states（不断丰富的上下文语义表示）               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  RMSNorm（最终层归一化）                                      │
│  输出: final hidden_states（归一化后的最终隐藏状态）           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  LM Head（语言模型头，线性层映射到词表大小）                   │
│  输出: logits (shape: [num_tokens, vocab_size])              │
│  说明: 每个 token 位置对词表中所有 token 的原始预测分数        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  LogitsProcessor（logits 处理器）                             │
│  输出: LogitsProcessorOutput（结构化输出对象）                │
│  包含:                                                        │
│    - next_token_logits（下一 token 的原始 logits）            │
│    - next_token_logprobs（采样后的对数概率）                  │
│    - input_token_logprobs（输入 token 的概率，可选）          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Sampler（采样器根据概率分布采样）                            │
│  采样策略: greedy / top-k / top-p / temperature             │
│  输出: 输出 token（选中的下一个 token id）                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Detokenize（分词器反向转换）                                 │
│  输出: 文本输出（最终生成的自然语言文本）                      │
└─────────────────────────────────────────────────────────────┘
```

一句话总结：

用户文本 → Tokenize → input_ids → Embedding → 初始 hidden_states → N 层 Decoder → 丰富后的 hidden_states → RMSNorm → 归一化后的 hidden_states → LM Head → logits → LogitsProcessor → LogitsProcessorOutput → Sampler → 输出 token → Detokenize → 生成文本


具体举例：
``` shell
用户文本 "今天天气"
    ↓
┌─────────────────────────────────────────────────────────┐
│ 流动的数据: input_ids (整数张量)                          │
│ 内容: [1001, 1002, 1003, 1004]  (token ids)             │
│ 形状: [batch_size=1, seq_len=4]                         │
│ 类型: torch.int64                                        │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 流动的数据: hidden_states (浮点张量)                      │
│ 内容: 每个 token 的语义向量表示                           │
│ 形状: [num_tokens=4, hidden_size=4096]                  │
│ 类型: torch.float16 / bfloat16                          │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 流动的数据: hidden_states (浮点张量，逐层更新)            │
│ 内容: 经过注意力层和 MLP 层后的语义表示                   │
│ 形状: [num_tokens=4, hidden_size=4096]                  │
│ 类型: torch.float16 / bfloat16                          │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 流动的数据: logits (浮点张量)                             │
│ 内容: 词表大小的预测分数                                  │
│ 形状: [num_tokens=4, vocab_size=32000]                  │
│ 类型: torch.float32                                      │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 流动的数据: output_token_ids (整数张量)                   │
│ 内容: 采样选中的 token id                                 │
│ 形状: [batch_size=1]                                     │
│ 类型: torch.int64                                        │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 流动的数据: 生成文本 (字符串)                             │
│ 内容: "很好"                                             │
│ 类型: str                                                │
└─────────────────────────────────────────────────────────┘

```

LogitsProcessor / Sampler vs Softmax 的对比:

SGLang 的 LogitsProcessor + Sampler 是 Softmax 的工程化扩展，提供了更丰富的采样控制和额外功能。

标准transformer:
```
hidden_states → LM Head → logits → Softmax → probs → argmax/sample → next_token
```

SGlang
```
hidden_states → LM Head → logits 
    ↓
[LogitsProcessor]
    - 应用惩罚/温度
    - 计算 logprobs (softmax → log)
    - 返回 LogitsProcessorOutput
    ↓
[Sampler]
    - 根据采样策略选择 token
    - 支持 greedy/top-k/top-p/temperature
    ↓
next_token
```

## PP

PP 模式下，中间 rank 不直接输出最终结果，而是把中间状态传给下一个 rank 继续计算。

## Forward 返回值三种场景

| 场景 | 条件 | 返回类型 | 用途 |
|------|------|---------|------|
| **普通推理** | 单卡/TP/DP，无辅助状态 | `torch.Tensor` | 标准 LLM 推理 |
| **推测解码** | `capture_aux_hidden_states=True` | `Tuple[Tensor, List[Tensor]]` | EAGLE3 draft model |
| **PP 中间 rank** | `not pp_group.is_last_rank` | `PPProxyTensors` | 流水线并行传递中间结果 |

## 如何新增模型实现

https://deepwiki.com/mingfeima/sglang/14.3-adding-new-models
