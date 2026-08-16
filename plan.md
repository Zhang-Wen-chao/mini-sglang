# mini-sglang 开发计划

用最少的纯 PyTorch 代码复现 SGLang 的核心思想：Radix Cache + 连续批处理调度器 + 最小推理引擎。
风格与 `mini-megatron` / `mini-deepspeed` 一致：独立仓库、CPU 单测、~800 行核心、无自定义 CUDA kernel。

## 交付物

- `mini_sglang/`：核心库（纯 torch + 标准库）
- `tests/`：CPU 单元测试（pytest）
- `examples/chat.py`：端到端可运行演示（可加载 HF 小模型权重）
- `README.md` / `plan.md` / `AGENTS.md`

## 设计原则

- 核心零外部依赖（除 torch）；HF transformers 只作为可选演示依赖，不进入核心。
- 复刻 SGLang 的抽象边界，但每个组件压缩到一屏可读：
  - radix cache 用区间 key（`(start, length)` 指向共享 token 数组）+ 节点分裂，对应
    `sglang/srt/mem_cache/radix_cache.py`。
  - KV 块按逻辑 token 位置映射物理块，块表（block table）显式持有，对应
    `sglang/srt/mem_cache/block_allocator.py` 与 `req_to_token_pool`。
  - 调度器 = FCFS 等待队列 + 内存预算 + radix 命中 + 抢占（recompute），对应
    `sglang/srt/managers/scheduler.py`。
- 不做：自定义 kernel、PagedAttention、异步引擎、speculative decoding、与 SGLang 逐位对齐。

## Phase 1：Radix Cache（字典树 + 引用计数 + LRU 驱逐）

- [x] 区间 key 的字典树（节点存 `(start, length)`，共享 token 数组）
- [x] `match_prefix`：最长公共前缀匹配，命中中途分裂节点
- [x] `insert`：插入新序列，必要时分裂共享节点
- [x] `inc_ref_count` / `dec_ref_count`：路径引用计数
- [x] `evict`：只驱逐 `ref_count == 0` 的叶子，LRU 序（last_access_time）
- [x] `total_len` / `pretty_print` / 命中统计
- [x] CPU 单测：`tests/test_radix_cache.py`

## Phase 2：连续批处理调度器（请求队列 + 抢占 + KV 块管理）

- [x] `kv_pool.py`：物理块分配器（free 列表、块表、块已写长度、K/V 张量池）
- [x] `scheduler.py`：
  - [x] 等待队列（FCFS）与 running 上限（模拟显存预算）
  - [x] prefill 时 `match_prefix` 复用前缀块（引用计数 +1）
  - [x] 内存不足：先 evict radix cache，再抢占（preempt，recompute 模式）
  - [x] 块完成规则：写满的块才进 radix cache；末块不完整留在请求上
  - [x] decode 步进、EOS / max_len 结束
- [x] CPU 单测：`tests/test_scheduler.py`（用确定性伪模型，不依赖真实模型）

## Phase 3：最小推理引擎（tokenize → prefill → decode → 返回）

- [x] `model.py`：Llama 风格最小模型（RMSNorm + RoPE + MHA + MLP），KV 读写走池子
- [x] `tokenizer.py`：零依赖 TinyTokenizer（byte 级）；可插拔 HF tokenizer
- [x] `engine.py`：连续批处理循环（调度 → 前向 → 采样 → 推进），采样 + EOS
- [x] `examples/chat.py`：端到端演示；GPU 主机上加载真实小模型权重（llama-68m）验证
- [x] CPU 单测：`tests/test_engine.py`（toy 随机模型 + TinyTokenizer）
- [x] GPU 验证：真实模型生成 + radix cache 命中可观测（73-token 共享前缀命中 64-token 块）

## 收尾

- [x] README：架构、快速开始、与 SGLang 的逐模块对应、与真实实现的差距
- [x] 与独立基线对比：HF fp32 逐字节一致 4/4；正版 SGLang 0.5.17 bf16 token 级一致 4/4
- [x] 吞吐参考记录（17–35× 差距，环境绑定，非基准声明）
- [x] 公开发布前清理检查（无凭据 / 无本机路径 / 无内部标识）
