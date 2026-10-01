# 与 CliffCompaction 论文对齐的实验设置

论文:*CliffCompaction: Cost-Efficient Compaction for Long-Horizon Coding Agents*(Nguyen, Cho, Chen, Dettmers;CMU / Bosch;arXiv 2609.26779,2026-09-22)。
下面只写论文里能查到的内容,页内没写明的标"未说明"。表号是论文里的表号。

## 1. 论文的实验设置

### 1.1 方法本身
- 阈值触发:上下文增长到阈值 B 才压缩,其余时间不动前缀(保缓存)。
- 压缩内容:工具结果 >500 字符整条丢弃,≤500 保留;工具调用只留签名(前 150 字符;OpenHands 里 command ≤120、old_str ≤60);assistant 的 thinking/analysis 截到 300 字符;system prompt 和第一条 user(任务)全文保留;最近 K 个 turn 原样保留(K 的取值正文未说明,开源代码默认 3)。
- "Cliff by cliff":每次压缩只处理上一次压缩之后的新增部分,丢弃上一次的压缩块,不嵌套。
- 不调用额外 LLM。

### 1.2 数据集与 harness(第 3.1 节)
| 基准 | harness | 模型 | 阈值 / 对照 |
|---|---|---|---|
| SWE-bench Verified | mini-swe-agent(追加式历史,无原生压缩) | Kimi K2.6 / K2.5,GLM 5.1 / 5 Turbo / 5 / 4.7 Flash | 全上下文 vs Cliff B∈{32K,16K,8K} |
| SWE-bench Verified | OpenHands(换掉原生 condenser) | Kimi K2.6,GLM 5.1 | 全上下文(原生 condenser 开)vs Cliff 32K |
| Terminal-Bench 2.0(89 题) | Terminus-2(换掉内置摘要) | Kimi K2.6,GLM 5.1 | 全上下文 vs Terminus-2 LLM 摘要 vs Cliff,B∈{32K,16K,8K} |
| Terminal-Bench 2.1 | **Claude Code**(API 代理) | GLM 5.3 Flash | Claude Code 原生 auto-compaction(匹配平均峰值 ~45K)vs Cliff(~45K)vs 默认 200K |
| KernelBench L3(50 题) | OpenHands | Kimi K2.7 / K2.6,GPT-5-mini | 不压缩 256K vs Cliff 128K/200K,200/400 步 |
| 测试时扩展(第 4 节) | Terminus-2 / mini-swe-agent | Kimi K2.6,GLM 5.1 | k 次 rollout + SGV 选择器 |

- 阈值集合:B ∈ {45K, 32K, 16K, 8K}。
- 具体题目子集、种子数、步数上限、超时、温度:**正文未说明**(表中 ±x 是多次运行的标准差,次数未写)。
- 模型价格(表 9,USD/百万 token,输入 / 缓存读 / 输出):Kimi K2.6 0.95/0.16/4.00,K2.7 0.95/0.19/4.00,K2.5 0.60/0.10/3.00;GLM 5.1 1.40/0.26/4.40,GLM 5.3 Flash 0.15/0.03/0.50。

### 1.3 对比对象(基线)
1. 全上下文:各 harness 的默认行为(无压缩,或只在窗口满时压缩)。
2. Terminus-2 内置 LLM 摘要(TB2.0,同阈值)。
3. Claude Code 原生 auto-compaction(TB2.1,匹配 ~45K 峰值),以及它的默认 200K。
4. 滑动窗口:保留固定 system/task 前缀加最近 turn,每步移动。
5. LLM 摘要:Claude Code 的摘要 prompt。
6. 摘要 + microcompaction:复刻 Claude Code,清空旧工具观察内容但保留调用和最近 5 条观察;触发量 τ=20,000(128K 预算)/4,000(16K)。
7. OpenHands 原生 condenser。
第 4–7 项在第 6 节、表 6(SWE 16K + KernelBench 128K,400 步)。

### 1.4 指标与成本口径
- 指标:% Resolved ±std,平均每实例成本;KernelBench 用几何平均加速比;另有步数、重读次数(图 6)、缓存命中率(表 8)。
- **成本按"完美缓存"重算(附录 A.1)**,不用厂商返回的 cached_tokens:第 1 次调用全部算未缓存;第 k 次若 P_k ≥ P_{k-1},则 P_{k-1} 缓存、其余未缓存;若 P_k < P_{k-1} 视为压缩,P_k 全部未缓存;输出按每次调用计价。同时另报真实计费和缓存命中率(表 8)。
- 主要数字(用于对照):
  - SWE-bench,mini-swe-agent,Kimi K2.6:全上下文 73.87% / $0.19,32K 73.27% / $0.18,16K 71.87% / $0.17,8K 67.60% / $0.18。GLM 5.1:71.40% / $0.25,68.80% / $0.20,69.33% / $0.15,65.13% / $0.13。
  - TB2.0,Terminus-2,Kimi K2.6:全上下文 59.16% / $0.40;摘要 32K/16K/8K = 58.36/55.45/42.97%,$0.24/$0.26/$0.60;Cliff = 61.42/61.42/50.00%,$0.24/$0.19/$0.25。
  - TB2.1,Claude Code,GLM 5.3 Flash:200K 摘要 73.03% / $0.21;45K 摘要 70.97% / $0.14;45K Cliff 76.69% / $0.16。
  - 表 6(SWE 16K,Kimi K2.7):滑窗 72.80% / $0.26(+9%),摘要 70.27% / $0.18(−23%),摘要+micro 71.00% / $0.20(−17%),Cliff 71.33% / $0.19(−21%)。
- 论文自己承认的限制(第 8 节):harness 的 system prompt 和工具定义占固定预算,所以各 harness 的最小可用阈值不同。**这正是 Claude Code 上 16K/8K 不可行的原因**(我们实测固定开销约 18K)。

## 2. 我们现在的设置与论文的差距

| 维度 | 论文 | 我们 | 状态 |
|---|---|---|---|
| 基准 | SWE-bench Verified(子集未说明)、TB2.0 89 题、TB2.1 | SWE-bench Verified 19 题(Epoch arm64 镜像),TB2 只跑过 8–10 题 | 子集小,标准误约 11 个点 |
| harness | mini-swe-agent / OpenHands / Terminus-2 / Claude Code | Terminus-2 + Claude Code | Terminus-2 与 Claude Code 可对;SWE 上论文没跑这两个 harness,TB 才是 |
| 模型 | Kimi K2.x、GLM 5.x | Qwen3.5-397B / 9B、Qwen3.6-35B-A3B(Tinker,64K 窗口) | 无法一致,取 35B-A3B 作"便宜开源模型"对位 |
| 全上下文 | 200K–256K | **≤56–64K**(Tinker 窗口) | 无法一致,"全上下文"只能是 ~56K |
| 阈值 | 45K / 32K / 16K / 8K | Terminus-2:16K、32K、EOQ;Claude Code:待定 | 缺 8K、45K |
| 压缩参数 | 工具结果 500、签名 150、thought 300、K 未说明 | 官方默认:500、150、**thought 不限(0)**、K=3 | **thought 截断 300 我们没开** |
| 对照 | 滑窗、LLM 摘要、摘要+micro、原生压缩 | Terminus-2 原生摘要(harness 默认);滑窗配置(`swe_ctx_slide`)已有;无摘要+micro | 见第 3 节 |
| 成本 | 完美缓存重算 + 真实计费 | 按服务商返回的缓存用量计价 | 需补"完美缓存"口径 |
| 重复 | ± 标准差 | 单次运行 | 需要多种子 |
| EOQ(我们自己的) | 无 | 有 | 论文之外的贡献,与论文设置并列 |

## 3. 对齐后的验证实验(建议)

原则:每个实验都能和论文的某一行直接对应,模型和窗口的差异在表里注明,不假装一致。

### A. Claude Code,对应论文表 2 右侧(TB2.1 / Claude Code)
- 任务:先用 SWE-bench Verified 19 题(镜像和基础设施已就绪);论文用的是 TB2.1,TB 的 Claude Code 版本之后再加。
- 模型:Qwen3.6-35B-A3B(mid)。
- 对照组:
  1. Claude Code 原生 auto-compaction,匹配平均峰值 ~45K(现 `CLAUDE_CODE_AUTOCOMPACT_PCT_OVERRIDE=25`,需要用实测峰值校准);
  2. Cliff **45K**(论文匹配点);
  3. Cliff **32K**(固定开销 ~18K,实际历史预算 ~14K);
  4. EOQ v1(我们的规则);
  5. "默认 200K":在 Tinker 上不可行,最接近的是不外加压缩、窗口顶满 ~56K。
- 16K、8K 不跑:固定开销 ~18K,阈值低于它会每次请求都压缩。论文第 8 节也承认这一点。
- 压缩参数:`CLIFF_THOUGHT_MAX_CHARS=300`、`CLIFF_THINKING_MAX_CHARS=300`,其余用官方默认。
- 成本口径:报完美缓存重算和真实计费两种。

### B. Terminus-2,对应论文表 2 左侧(TB2.0 / Terminus-2)和表 3
- 任务:SWE-bench 19 题(已有基线);TB2 取子集。
- 对照组:Terminus-2 原生摘要 {32K, 16K, 8K}(用 `STAYSWITCH_MAX_INPUT_TOKENS=B+8K` 触发)、Cliff {32K, 16K, 8K}、滑窗、EOQ。
- 已有结果可复用的有:`swe_mid`、`swe_mid_cliff16`、`swe_mid_cliff_eoq`;`swe_mid_cliff32` 被中断,要重跑;缺 8K、摘要对照、thought=300 的版本。

### C. 重复次数
- 论文报 ±std,我们每个设置至少 3 个种子才能报标准差;19 题单次的标准误约 11 个点,不足以分辨论文里 1–2 个点的差异。
- 阶段 1 单种子跑全矩阵,阶段 2 对关键对比(Cliff 45K vs 原生 vs EOQ)加种子。

### D. 成本估算(用我们已有数据)
- mid 模型每条轨迹:Terminus-2 约 $0.15–0.20,Claude Code 约 $0.18(10 条)。
- 19 题一个设置约 $3–4,3 个种子约 $10–12。
- 当前台账已花约 $171,上限 $245,剩余约 $74。矩阵 A(4–5 组)× 3 种子 ≈ $50;B 另算。需要你确认预算。
- TB2 单条轨迹贵很多(strong $2.56,weak $0.73),全 89 题多设置不可行,只建议子集。

## 4. 需要的代码/配置改动
1. `CLIFF_THOUGHT_MAX_CHARS=300`、`CLIFF_THINKING_MAX_CHARS=300`:由 `run_cliff.sh` 继承环境变量,不用改脚本。
2. 新增 `cc_mid_cliff45.toml`、`cc_mid_cliff32.toml`;之前建的 28K/40K 配置作废。
3. 新增"完美缓存"成本脚本:按附录 A.1 从每条轨迹的 prompt 长度序列重算成本,并入 `summarize_runs.py` 的输出(我来写,不改别人的脚本,先放新文件)。
4. Claude Code 原生 auto-compaction 的峰值校准:跑一遍后取平均峰值,调 `CLAUDE_CODE_AUTOCOMPACT_PCT_OVERRIDE` 让它落在 ~45K。

## 5. 待确认
1. 预算:是否按上面的 A 矩阵 × 3 种子(约 $50)走?
2. 是否要把 TB2.1(或 TB2.0 子集)加进来作为与论文表 2 完全同基准的一组?需要 TB 镜像预构建。
3. 模型:接受 Qwen3.6-35B-A3B 作为 Kimi / GLM 的对位吗?(公司网络内的 DeepSeek V4 网关可作第二个模型,但窗口和缓存语义不同。)
