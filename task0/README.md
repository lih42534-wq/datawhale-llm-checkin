微信群昵称：旅行包  
环境：Mac，本次使用本地 CPU  
系统：macOS 26.5.1，Python 3.11.16，PyTorch 2.14.1  
AI 占比 vs 人工占比：本次代码补全、运行和笔记整理主要由 AI 辅助完成，人工确定任务范围和提交方式。  
任务项：**4.1 + 4.2**

### 4.1 最小打卡

**（1）推理模型架构的组成**

我把常见的 Decoder-only Transformer 理解成三个部分：Embedding 把 token ID 变成向量；中间多层 Transformer Block 处理上下文，每层包含因果自注意力、前馈网络、归一化和残差连接，并通过 RoPE 等方式引入位置信息；最后由 LM Head 把隐藏表示转成词表上的分数，再按生成策略选择下一个 token。

推理时还会保存历史 token 的 Key/Value，也就是 KV Cache。它是在 Prefill 时建立、Decode 时读取和追加的状态。MHA、GQA、MLA 会改变缓存的组织和容量，Dense、MoE 则会改变每个 token 的计算路径。

**（2）一个请求的过程，Prefill、Decode、TTFT、TPOT**

一次请求可以按这条线理解：

`请求到达 → 排队 → 准入与输入准备 → Prefill → 首个 token → Decode → 结束`

输入准备包括分词等操作。Prefill 处理输入序列，建立 KV Cache，得到预测首个 token 所需的结果；Decode 利用已有缓存逐步生成后续 token，同时更新缓存，直到遇到结束符、停止条件或长度上限。取消、超时和失败也属于请求结果，需要记录发生在哪个阶段。

- **TTFT**：从请求到达到首个输出 token 的时间，也就是“等多久才开始出字”。在本题里，等于排队 + 输入准备 + Prefill 的时间。
- **TPOT**：首个 token 之后，相邻输出 token 间隔的平均值，也就是“后面平均多久出一个 token”。公式是 `(最后一个 token 时间 − 首 token 时间) / (输出 token 数 − 1)`。

没有首 token 时，TTFT 应保持 `None`；少于两个 token 时，TPOT 也应为 `None`，不能用 0 代替。

**（3）INF-REQUEST 请求生命周期代码运行**

代码部分借助 AI 补全了四处 TODO：合法状态迁移、TPOT 计算、阶段占比归因和工作负载统计分母。在 VS Code 中用本地 Python 环境运行，原题 **6 组测试全部通过**，另外补了单 token、空工作负载、零时长归因 3 项边界检查，并用 PyTorch tensor 交叉验证指标。

我用测试中的时间戳核对了一遍：0 ms 到达，2 ms 准入，3 ms 输入就绪，7、9、11 ms 输出三个 token，12 ms 结束。这样 TTFT 是 7 ms，TPOT 是 `(11−7)/2=2 ms/token`，E2E 是 12 ms。这里是模拟事件的计算结果，不是真实模型的测速。

一个容易混淆的地方是：首 token 到请求结束用了 5 ms，但计算 TPOT 应使用首末 token 的跨度 4 ms。另一个是统计分母：一次完成加一次首 token 前取消，成功率是 50%，但 TTFT 只有 1 个有效样本，取消请求不能当作 TTFT=0。

![VS Code 中的 Python 代码、PyTorch 验证与测试通过结果](https://raw.githubusercontent.com/lih42534-wq/datawhale-llm-checkin/main/task0/vscode-pytorch-run.jpg)

[Python 代码](https://github.com/lih42534-wq/datawhale-llm-checkin/blob/main/task0/task0_request_lifecycle.py) · [已运行 Notebook](https://github.com/lih42534-wq/datawhale-llm-checkin/blob/main/task0/task0_executed.ipynb)

### 4.2 模型、Tokenizer、推理框架、GPU、API 服务的关系

我的理解是：API 服务接收请求，Tokenizer 把文本变成模型匹配的 token ID；推理框架负责加载模型、调度请求、组织批处理和管理 KV Cache，在 GPU 等硬件上执行模型计算；模型给出下一个 token 的预测分数，按生成策略选择 token，再由 Tokenizer 转回文本，通过 API 返回。

模型是网络结构和权重，Tokenizer 负责文本转换，推理框架负责组织执行，GPU 提供算力和显存，API 提供外部调用入口。本次生命周期练习在 CPU 上即可完成。

当前卡点：机制测试已通过，真实模型服务的启动与性能验证留到后续继续做，本次未选 4.3。

学习来源：DataWhale 社区 [llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)；[01 请求链路与指标](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/inference_optimization/01_request_path_and_metrics.md)；[INF-REQUEST 请求生命周期](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.ipynb)。
