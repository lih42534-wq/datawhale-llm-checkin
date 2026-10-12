微信群昵称：旅行包  
GitHub ID：lih42534-wq  
环境：Mac（Apple Silicon），本次使用 CPU，无 NVIDIA GPU  
系统：macOS 26.5.1；Python 3.11.16；PyTorch 2.14.1  
AI 占比 vs 人工占比：100:0（按本次代码实现、运行与文字初稿产出计；本人提出需求并确认提交）。  
任务项：**4.1 最小打卡 + 4.2 增项 1**

### 4.1（1）推理模型架构的组成

以常见的 Decoder-only 自回归 Transformer 为例，模型把已有上下文变成“下一个 token 的预测分数”，再按生成策略选择 token，接回上下文继续预测。

- **Embedding**：把 token ID 映射成向量。
- **多层 Transformer Block**：因果自注意力聚合当前及此前位置的信息；FFN/MLP 做非线性变换；归一化与残差连接配合完成深层计算。位置信息可通过 RoPE 等机制引入。
- **最终归一化与 LM Head**：将隐藏表示映射到词表维度，产生 logits。

推理时还要维护 **KV Cache**，复用历史位置的 Key/Value，减少重复计算。注意力采用 MHA、GQA 或 MLA，会改变历史状态的组织和容量；前馈部分采用 Dense 或 MoE，会改变每个 token 激活的计算路径。KV Cache 是运行状态，贯穿 Prefill 和 Decode。

### 4.1（2）请求过程、Prefill / Decode、TTFT / TPOT

一个请求可按以下顺序观察：

`到达 → 排队 → 准入 → 输入准备 → Prefill → 首 token → Decode → 终止与资源回收`

输入准备包括分词、应用聊天模板等。正常结束可由 EOS、停止条件或长度上限触发；取消、超时、失败也要记录发生阶段和已输出 token 数。

**Prefill** 处理输入序列，建立初始 KV Cache，并提供预测首个输出 token 所需的结果。**Decode** 在普通自回归生成中利用历史缓存，每轮处理新 token、预测下一个 token，并追加缓存。

**TTFT（Time To First Token）**回答“等多久开始输出”：

`TTFT = 首 token 时刻 − 请求到达时刻`

按本题事件边界，也等于排队 + 输入准备 + Prefill 的时间。客户端计时还可能包含网络传输，因此比较时必须统一测量位置。

**TPOT（Time Per Output Token）**回答“开始输出后，平均多久出一个 token”：

`TPOT = (最后一个 token 时刻 − 首 token 时刻) / (输出 token 数 − 1)`，要求至少两个输出 token。

它是耗时指标，常用 ms/token，不能直接等同于整个服务的吞吐量。没有首 token 时 TTFT 为 `None`；不足两个 token 时 TPOT 为 `None`，缺失值不能用 0 代替。

### 4.1（3）代码实现与真实运行结果

补全了指定 INF-REQUEST notebook 的四处 TODO：合法状态迁移、相邻 token 间隔、阶段占比归因、工作负载统计分母。保留原题六组测试，并额外检查单 token、空工作负载和零时长归因。

在本地 CPU 实际执行，**原题 6/6 测试通过，3/3 边界检查通过**。另外使用 **PyTorch CPU tensor** 计算 `torch.diff(token_times).mean()` 与阶段占比，和原实现交叉核对，全部通过。使用 VS Code 展示 Python 文件与终端结果。

这次是**合成事件回放练习**，把事件时间戳统一解释为毫秒，以下数值不代表真实大模型推理速度：

| 项目 | 结果 |
| --- | --- |
| 排队 / 输入准备 / Prefill | 2 / 1 / 4 ms |
| 输出 token 时间戳 | 7、9、11 ms |
| TTFT | 7 ms |
| TPOT | 2 ms/token |
| 请求终止时刻 / E2E | 12 ms / 12 ms |
| 阶段归因 | mixed，最大占比 5/12≈41.67%，小于题设 45% 阈值 |

最后一个 token 在 11 ms 输出，请求在 12 ms 终止，因此首 token 到终止的 5 ms 不能代替 TPOT 公式里的 token 跨度 4 ms。

对“一次完成 + 一次首 token 前取消”的工作负载，成功率为 1/2；TTFT 只有 1 个有效样本，均值为 7 ms。取消请求仍进入成功率分母，但不能伪造一个 TTFT=0 混入平均值。

**VS Code 实际运行截图：**

![旅行包：VS Code 中的 PyTorch 计算及全部测试通过](https://raw.githubusercontent.com/lih42534-wq/datawhale-llm-checkin/main/task0/vscode-pytorch-run.jpg)

[Python 源码](https://github.com/lih42534-wq/datawhale-llm-checkin/blob/main/task0/task0_request_lifecycle.py) · [已执行 Notebook](https://github.com/lih42534-wq/datawhale-llm-checkin/blob/main/task0/task0_executed.ipynb) · [运行日志](https://github.com/lih42534-wq/datawhale-llm-checkin/blob/main/task0/run-log.txt)

### 4.2 模型、Tokenizer、推理框架、GPU、API 服务之间的关系

可以沿调用链理解：API 服务接收文本和生成参数；Tokenizer 将文本转为模型匹配的 token ID；推理框架加载模型、组织调度与批处理、管理 KV Cache，并在 GPU 或其他设备上执行模型；模型输出 logits，生成策略选择 token；Tokenizer 解码为文本，再由 API 流式返回。

模型由网络结构和权重组成；Tokenizer 负责文本与 token ID 的转换；推理框架负责组织执行；GPU 提供并行计算和显存；API 服务提供对外接口。GPU 是常见硬件选择，本题的状态机与指标练习可直接在 CPU 上完成。

当前卡点：本次机制练习已通过；尚未启动真实大模型推理服务，未申报 4.3。后续比较真实后端时，需要固定模型/分词器版本、输入输出长度、并发、到达方式、缓存策略与数据类型，并保留失败请求。

学习来源：**DataWhale 社区** [llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)；[专题正文 · 01 请求链路与指标](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/inference_optimization/01_request_path_and_metrics.md)；[Part 02 · INF-REQUEST 请求生命周期](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.ipynb)。
