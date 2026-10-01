# StaySwitch 使用手册

StaySwitch 是一套研究 coding agent 成本的实验工具。它在 agent 和模型之间放一个代理(proxy),可以：

- 按策略把每一次模型调用路由到不同模型;
- 在发给模型之前压缩历史上下文(自己实现的压缩器，或者串联 CliffCompaction 官方实现);
- 把每次调用的 token 拆成"新输入 / 缓存读 / 缓存写 / 输出"四类记账，并按价目表算钱;
- 回放已有轨迹的前缀，从中间分叉(闭环分叉);
- 按轨迹和总量设花费上限。

任务通过 [Harbor](https://hub.harborframework.com) 运行(Terminal-Bench 2、SWE-bench Verified 等),agent 默认用 terminus-2。

---

## 1. 整体结构

```
agent (Harbor 里的 terminus-2)
   │  OpenAI 格式请求，带 X-Session-ID
   ▼
[可选] CliffCompaction 官方 proxy   ← 只在 run_cliff.sh 里串联，负责压缩
   ▼
StaySwitch proxy (LiteLLM + stayswitch_hook)   ← 路由、压缩、记账、预算、回放
   ▼
模型服务(Tinker / 私有网关 / mock)
```

| 路径 | 作用 |
|---|---|
| `src/stayswitch/` | 核心库：计价、缓存语义与账本、成本模型、会话、策略、信号、压缩、预算、记账 |
| `proxy/stayswitch_hook.py` | LiteLLM 回调：每次调用前做路由和压缩，调用后写日志 |
| `proxy/litellm_config.yaml` | 模型池(`strong` / `mid` / `weak` / `ds-*` / `mock-*`) |
| `configs/*.toml` | 每个实验一份配置;`prices.toml` 是价目表 |
| `scripts/` | 运行、建镜像、汇总等脚本(见第 5、6 节) |
| `runs/<run_id>/calls.jsonl` | 每次模型调用的日志(不进 git) |
| `jobs/<run_id>/` | Harbor 的 trial 结果，含 reward 和轨迹(不进 git) |
| `data/` | 导出的数据集任务文件(不进 git) |

---

## 2. 安装

需要 [uv](https://docs.astral.sh/uv/)、Docker(本机用 colima)、Python 3.12。

```bash
cd stayswitch
uv sync                      # 主环境:Harbor + stayswitch + CliffCompaction(依赖组 cliff)
(cd proxy && uv sync)        # proxy 环境:LiteLLM proxy + stayswitch(可编辑安装)+ torch(RouteLLM 用)
uv run pytest -q             # 单元测试，应全部通过
```

为什么有两个环境:LiteLLM proxy 和 Harbor 依赖的 `rich` 版本冲突，所以分开装。两边都以可编辑方式安装了 `stayswitch`,改代码后不用重装。

### 2.1 密钥

在 `stayswitch/.env` 里写(这个文件已被 gitignore,建议权限设为 600):

```
TINKER_API_KEY=...           # strong / mid / weak 模型池
GATEWAY_TOKEN=...            # 可选:ds-* 模型池(私有 Anthropic 兼容网关)
GATEWAY_BASE=...
GATEWAY_LIMIT_PRICE=...
```

`scripts/start_proxy.sh` 启动时会自动读取它。

### 2.2 本机网络和 Docker(在这台 Mac 上踩过的坑)

| 问题 | 处理 |
|---|---|
| colima 资源不够 | `colima stop && colima start --cpu 6 --memory 12` |
| `docker compose` 不存在 | `brew install docker-compose`,并在 `~/.docker/config.json` 的 `cliPluginsExtraDirs` 里加上 `/opt/homebrew/lib/docker/cli-plugins` |
| Docker Hub 拉不下来 | `~/.colima/default/colima.yaml` 的 `docker.registry-mirrors` 配置镜像源(docker.m.daocloud.io、mirror.gcr.io)。第三方镜像镜像源不放行，用第 4 节的方法本地构建 |
| 容器里访问 GitHub 卡死 | `~/.docker/config.json` 里配置 `proxies.default`,让容器和构建都走本机代理 `host.docker.internal:7897`。本机代理客户端关掉时，容器访问外网会失败 |
| 容器里 apt 很慢 | 已自动处理：自定义 agent 启动前会把 apt 源换成清华镜像(`STAYSWITCH_APT_MIRROR`) |
| 验证脚本从 GitHub 下载 uv 失败 | 已自动处理:`run_harbor.sh` 会在本机 8765 端口启动 `gh_release_cache.py`,容器通过它下载 |
| npm 默认源是内网，连不上 | 临时加 `--registry https://registry.npmjs.org` |

---

## 3. 模型池和价格

`proxy/litellm_config.yaml` 里定义的别名(配置文件里写的就是这些名字):

| 别名 | 实际模型 | 说明 |
|---|---|---|
| `strong` | Qwen/Qwen3.5-397B-A17B(Tinker) | 默认开启思考 |
| `mid` | Qwen/Qwen3.6-35B-A3B(Tinker) | 单价比 9B 还低;SWE 上解决率与 strong 相当 |
| `weak` | Qwen/Qwen3.5-9B(Tinker) | 步数膨胀严重，一般不推荐 |
| `ds-strong` / `ds-weak` | DeepSeek V4 pro / flash(私有网关) | 需要 `GATEWAY_*` |
| `mock-strong` / `mock-weak` | 不调用任何服务 | 离线测试用，花费不计入预算 |

`configs/prices.toml` 是每百万 token 的价格(USD):`input`、`output`、`cache_read`、`cache_write`,也可以用 `cache_read_mult` / `cache_write_mult` 按输入价的倍数给。Tinker 的规则是：缓存命中按 0.2 倍计费，没有写入溢价。所有报告里的花费，都是用日志里的 token 数按这张表重新算出来的，不是服务商账单。

注意:Tinker 的 Cloudflare 会拦截某些默认 User-Agent(报错 1010),所以模型池里显式设置了 `User-Agent: stayswitch/0.1`。

---

## 4. 准备数据集和镜像

### 4.1 Terminal-Bench 2

TB2 任务的预构建镜像放在 Docker Hub 个人仓库里，镜像源不放行，所以在本地按任务自带的 Dockerfile 构建，并打上原来的 tag:

```bash
uv run harbor datasets download terminal-bench/terminal-bench-2 --cache
uv run python scripts/prebuild_images.py -j 3 --only build-pov-ray circuit-fibsqrt   # 不加 --only 就构建全部
```

构建时会把 apt 源换成清华镜像，失败自动重试 3 次。日志在 `runs/prebuild/`。和官方镜像相比，apt 源不同;没锁版本的包可能更新。写论文时要注明这两点。

### 4.2 SWE-bench Verified

Harbor 的任务基于 `swebench/sweb.eval.x86_64.*`(Docker Hub、仅 x86)。我们改用 Epoch AI 在 GHCR 上发布的 arm64 版本，并打上 Dockerfile 期望的 tag:

```bash
uv run harbor datasets download swe-bench/swe-bench-verified -o data     # 导出任务文件到 data/
uv run python scripts/swe_images.py select data/swe-bench-verified       # 分层抽样，写 configs/swe_tasks.txt
uv run python scripts/swe_images.py pull data/swe-bench-verified -j 8    # 在本机下载并导入 Docker
```

`pull` 调用 `scripts/fetch_image.py`:在本机用多连接、可断点续传的方式下载各层，按内容哈希去重缓存(`~/.cache/stayswitch/blobs`),然后 `docker load`。GHCR 很慢，直接 `docker pull` 会经常断线重来。同一仓库的任务共享底层，19 个任务去重后约 3.8 GB。

`select` 的规则：在"<15 分钟"和"15 分钟到 1 小时"两档里各抽 `--n-per-bucket` 个(默认 10),每个仓库最多 3 个，排除体积最大的 1/4 镜像。

---

## 5. 跑实验

### 5.1 一个配置跑一批任务

```bash
scripts/run_model.sh <dataset> <task_list> <config> [port=4001] [concurrency=2]

# 例:SWE-bench 上全程 mid,并发 3
scripts/run_model.sh swe-bench/swe-bench-verified configs/swe_tasks.txt configs/swe_mid.toml 4001 3
```

它会依次：启动 proxy → 运行 Harbor → 关闭 proxy → 打印汇总。run id 取自配置里的 `[run].id`,日志写到 `runs/<id>/`,trial 写到 `jobs/<id>/`。

### 5.2 串联 CliffCompaction

```bash
scripts/run_cliff.sh <task_list> <config> fixed <threshold_tokens>   # 官方实现，固定阈值
scripts/run_cliff.sh <task_list> <config> eoq   <extra_steps>        # 官方压缩机制 + EOQ 阈值

scripts/run_cliff.sh configs/swe_tasks.txt configs/swe_cliff16.toml fixed 16000
scripts/run_cliff.sh configs/swe_tasks.txt configs/swe_cliff_eoq.toml eoq 2
```

- 链路是 agent → CliffCompaction(端口 `CLIFF_PORT`,默认 8257)→ StaySwitch proxy(`ROUTER_PORT`,默认 4001)→ 模型。
- 会自动关掉 terminus-2 自带的摘要(`STAYSWITCH_MAX_INPUT_TOKENS=1000000`),压缩完全交给 CliffCompaction。
- 其他环境变量:`DATASET`、`KEEP_RECENT`(默认 3)、`CONC`(默认 3)。
- EOQ 模式下，每次请求算出的阈值记录在 `runs/<id>/cliff_thresholds.jsonl`;官方 proxy 的日志在 `runs/<id>/cliff.log`。
- CliffCompaction 锁定在 commit `b48d660`(见 `pyproject.toml` 的 `cliff` 依赖组)。

### 5.3 排队跑多个配置

```bash
nohup scripts/run_queue.sh swe-bench/swe-bench-verified configs/swe_tasks.txt \
  configs/swe_strong_lead.toml configs/swe_trigger.toml > runs/queue.log 2>&1 &
```

- 会先等当前的 Harbor 任务结束，然后一次只跑一个配置。
- 轮到某个配置时才读取它，所以排在后面的配置可以根据前面的结果再修改。
- 配置里含 `TODO` 字样的会被跳过。

### 5.4 TB2 上 strong 和 weak 并行跑基线

```bash
scripts/tb2_baseline.sh configs/tb2_tasks.txt 2
```

### 5.5 run_harbor.sh 的环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `STAYSWITCH_PORT` | 4000 | StaySwitch proxy 端口 |
| `STAYSWITCH_API_BASE` | `http://127.0.0.1:$STAYSWITCH_PORT` | agent 的 `api_base`(串联 CliffCompaction 时改为它的 `/v1`) |
| `STAYSWITCH_MAX_INPUT_TOKENS` | 56000 | 告诉 terminus-2 的上下文上限，它会在剩余约 8K 时自己做摘要 |
| `STAYSWITCH_AGENT` | `stayswitch.agents:StaySwitchTerminus` | 使用的 agent |
| `STAYSWITCH_APT_MIRROR` | 清华镜像 | 设为空字符串则不换源 |
| `VERIFIER_TIMEOUT_MULT` / `SETUP_TIMEOUT_MULT` | 5 | 验证 / agent 安装阶段的超时倍数 |
| `GH_CACHE_PORT` | 8765 | GitHub release 缓存服务的端口 |

其余参数会原样传给 `harbor run`,比如 `-i <org/task>`、`-l <n>`、`-n <并发>`。

---

## 6. 配置文件

```toml
[run]
id = "swe_cliff16"                 # 决定 runs/<id>/ 和 jobs/<id>/

[policy]                           # 路由策略，见 6.1
kind = "fixed"
model = "strong"

[context]                          # 可选：在 StaySwitch proxy 里压缩历史，见 6.2
mode = "budget"
budget = 16000
keep = 4

[budget]                           # 可选：花费上限
per_session_usd = 3.0              # 单条轨迹上限;超出后 proxy 替 agent 回复"任务完成"
total_usd = 245.0                  # 所有 runs/*/calls.jsonl 合计的上限(不含 mock 模型)

[cache]                            # 可选：供应商的缓存规则，见 6.3
preset = "tinker"

[prices]
path = "prices.toml"

[log]
path = "../runs/swe_cliff16/calls.jsonl"
```

相对路径都以配置文件所在目录为基准。

### 6.1 路由策略 `[policy]`

| `kind` | 参数 | 含义 |
|---|---|---|
| `fixed` | `model` | 全程用一个模型 |
| `random_step` | `weak`, `strong`, `p_weak`, `seed` | 每次调用独立地以概率 `p_weak` 用 weak |
| `random_segment` | `models`, `p_switch`, `min_stay`, `seed` | 随机切换，每次至少停留 `min_stay` 步 |
| `weak_first` | `weak`, `strong`, `k` | 前 `k` 次调用用 weak,之后一直用 strong(SWE-Router 的固定前缀形式) |
| `trigger_escalate` | `weak`, `strong`, `fail_streak`, `max_weak_steps` | 默认 weak;连续报错、重复动作或达到步数上限时永久升级(TACIT / ReDAct 风格) |
| `cost_escalate` | 同上，加 `value_usd`, `remaining_steps`, `out_tokens` | 触发条件和 `trigger_escalate` 相同，但只有升级的额外花费(切到 strong 的冷缓存溢价 + `remaining_steps` × 两模型每步差价，按 `[cache]` 语义算)不超过 `value_usd` 才真的升级；太贵则继续用 weak，下一次调用再算。刚压缩或摘要过时溢价为零。扫 `value_usd` 得到成本 / 解决率曲线。`note` 记录 `cost=` 或 `too_expensive=` |
| `strong_lead` | `weak`, `strong`, `lead_max`, `fail_streak`, `commit` | strong 定位并完成第一次修改，跑完测试后交给 weak;出问题时 strong 回来至少 `commit` 步 |
| `cost_lead` | 同 `strong_lead`，加 `value_usd`, `remaining_steps`, `out_tokens` | `strong_lead` 的两个切换都过成本门：strong 交给 weak 前，先算切到 weak 并最终切回的回本步数，超过 `remaining_steps` 就留在 strong(`note=hold`)；weak 召回 strong 前，算 strong 的缓存溢价(lead 阶段的前缀通常还在，所以只补 follow 那一段)加 `commit` 步差价，超过 `value_usd` 就不召回(`note=rescue_deferred`)，下一次再算 |
| `routellm` | `weak`, `strong`, `threshold`, `checkpoint` | RouteLLM BERT 路由器逐次打分;首次运行会从 HuggingFace 下载约 1.1GB 权重 |
| `fork` | `source_log`, `fork_step`, `option_model`, `base_model`, `horizon`, `check_divergence`, `source_sessions` | 闭环分叉，见第 8 节 |

terminus-2 做摘要时会用派生的会话 ID(`<id>-summarization-*`),这些调用会沿用所属轨迹当前的模型。

### 6.2 压缩 `[context]`

| `mode` | 参数 | 行为 |
|---|---|---|
| `evict` | `keep`, `chunk` | 保留最近 `keep` 轮;每满 `chunk` 轮整块淘汰一次(`chunk=1` 即滑动窗口，每步都会破坏缓存) |
| `budget` | `budget`, `keep`, `max_msg_chars` | 估计长度达到 `budget` 时一次性截断到"任务描述 + 最近 `keep` 轮",之后截断点保持不动 |
| `eoq` | `keep`, `extra_steps`, `out_tokens`, `ceiling` | 阈值按 EOQ 公式在线计算:L\* = L₀ + √(2·g·C / c_r) |

两条保护规则：超长消息只按内容本身截断，不看它在第几位，所以同一条消息每次发出的内容一致;两次压缩之间至少间隔 4 步的增长量，防止退化成滑动窗口。

做压缩时间实验，建议优先用第 5.2 节的 CliffCompaction 串联方式，它的压缩机制是公开实现，可比性更好。

### 6.3 缓存语义 `[cache]`

描述供应商怎样缓存和计费前缀。`preset` 可选 `table`(默认：持久缓存，缓存价格取 `prices.toml`)、`tinker`、`deepseek`、`anthropic`、`oblivious`(完全不考虑缓存)；其余字段可以逐个覆盖：

| 字段 | 含义 |
|---|---|
| `ttl_s` | 缓存存活时间(秒)；省略或 `None` 表示持久，`0` 表示从不缓存 |
| `read_mult` / `write_mult` | 缓存读 / 写价格相对输入价的倍数；不填则用价目表 |
| `min_prefix` | 短于此的前缀不缓存(Anthropic 为 1024) |
| `cross_model` | 假设不同模型可以共用缓存(第 12 节的上限估算) |

proxy 用它构造 `CostModel`，并在每条日志里写 `cache_pred`(按这套语义预测本次能读到的缓存 token 数)，和实际的 `usage.cache_read` 对照，可以检验语义是否符合供应商的真实行为。

---

## 7. 结果和分析

### 7.1 汇总

```bash
uv run python scripts/summarize_runs.py swe_strong swe_mid swe_cliff16
```

它会输出每个 run 的解决率、平均调用数、上下文长度、缓存命中率和花费，以及逐任务的对比表，并写入 `runs/summary_<runs>.json`。

- 同一个任务被跑了多次(比如重跑)时，只取最新的那次 trial。
- trial 和会话的对应关系，靠"任务 key + agent 执行时间窗口"确定。

### 7.2 调用日志字段(`runs/<id>/calls.jsonl`,每行一次调用)

| 字段 | 含义 |
|---|---|
| `session_id`, `task`, `step` | 轨迹 ID、任务 key(第一条用户消息的归一化哈希)、这条轨迹的第几次调用 |
| `model`, `prev_model`, `switched` | 本次用的模型、上一次的模型、是否发生切换 |
| `note` | 策略备注，如 `lead`、`rescue`、`replay`、`budget_stop`、`summary` |
| `replayed` | 是否为回放 / 预算截停的模拟回复(这类调用不计费) |
| `usage` | `fresh_input` / `cache_read` / `cache_write` / `output` 四类 token |
| `cache_pred`, `cache_semantics` | 按 `[cache]` 语义预测的缓存读 token 数(回放调用为空)，以及所用语义的名字 |
| `cost` | 按 `prices.toml` 算出的花费 |
| `n_messages`, `n_messages_sent`, `ctx_dropped_turns`, `ctx` | 收到的消息数、实际发出的消息数、压缩信息 |
| `input_hash`, `diverged_at` | 分叉时用来检查回放前缀是否一致 |
| `ok`, `error`, `latency_s`, `ts` | 是否成功、错误信息、耗时、时间戳 |

### 7.3 用不同缓存规则重新算钱

```python
from stayswitch.accounting import Call, reprice
from stayswitch.cache import ANTHROPIC, TINKER, CacheSemantics
from stayswitch.calllog import read_calls
from stayswitch.pricing import PriceTable

prices = PriceTable.load("configs/prices.toml")
calls = [Call.from_record(r) for r in read_calls("runs/swe_strong/calls.jsonl") if r.get("ok")]  # 按 session 分组后再传
bill = reprice(calls, prices, TINKER)                                  # 或 ANTHROPIC、CacheSemantics(ttl_s=600, ...)
bill.cache_aware, bill.cache_oblivious, bill.switches, bill.cold_calls
```

压缩或摘要之后 prompt 变短，所有模型的前缀缓存都会失效；`reprice` 通过日志里的 `ctx.compacted` 和"prompt 比上一次短"两个信号识别这一点。

重算 E0(proposal 第 11 节的表)用现成脚本，它对每条轨迹保持 token 序列不变、只改每步记在哪个模型上，再按 Tinker / DeepSeek / Anthropic 三种规则算钱，并把旧算法(不识别摘要和压缩)的数字放在方括号里对照:

```bash
uv run python scripts/e0_reprice.py tb2_strong                      # 默认 strong→weak，20 步分段，5 个种子
uv run python scripts/e0_reprice.py tb2_strong --weak mid --json runs/e0_tb2.json
```

切换的解析成本用 `CostModel`:

```python
from stayswitch.costmodel import CostModel
cm = CostModel(prices, TINKER)
cm.step_cost("strong", ctx_tokens=60_000, new_tokens=2_000, output_tokens=800)      # 缓存全热时一步的花费
cm.breakeven_steps("strong", "mid", ctx_tokens=60_000, new_tokens=2_000, output_tokens=800)  # 切过去几步能回本(含切回的代价)
cm.switch_premium("mid", state.cache, ctx_tokens=..., new_tokens=..., output_tokens=...)      # 在线：按会话账本算冷缓存溢价
```

### 7.4 当前花费

```bash
uv run python -c "from stayswitch.budget import spent_in_logs; print(spent_in_logs('runs/*/calls.jsonl'))"
```

---

## 8. 闭环分叉

1. 用 `fixed` 策略先跑一遍源轨迹，比如 `configs/swe_strong.toml`。
2. 写一个 `kind = "fork"` 的配置:
   - `source_log` 指向源轨迹的 `calls.jsonl`;
   - `fork_step`:在第几次调用分叉;
   - `option_model`:分叉后用哪个模型;
   - `horizon`:用多少步后切回 `base_model`,0 表示一直用到结束。
3. 用这个配置在同样的任务上再跑一遍。

分叉点之前,proxy 直接返回录下来的回复,agent 会重新执行同样的命令，环境也就随之重建，不需要容器快照。如果某一步的输入和源轨迹对不上(`check_divergence`),就提前在那一步开始分叉，并记下 `diverged_at`。计算哈希前会先把容器主机名归一化。

---

## 9. 离线测试

`mock-*` 模型不调用任何服务，可以用来检查 proxy 的行为:

```bash
scripts/start_proxy.sh configs/mock_fixed.toml 4011 &
(cd proxy && uv run python ../scripts/smoke_client.py 4011 6)
```

`configs/mock_*.toml` 分别覆盖了固定模型、分叉、预算截停、压缩、strong 领头策略和双 proxy 串联。

---

## 10. 常见问题

| 现象 | 原因和处理 |
|---|---|
| trial 报 `RewardFileNotFoundError` 或 `RuntimeError: Failed to start tmux` | 一般是基础设施问题。只重跑这几个任务：把任务名写进一个列表，用同一个配置再跑一次，汇总时会自动采用最新的 trial |
| 同一批任务在重复跑 | 检查是不是启动了两个 Harbor 任务(`pgrep -fl "harbor run"`)。要停掉某个 `run_model.sh` 而不影响共用的 proxy 时，用 `kill -9`,否则它退出时会顺手关掉 proxy |
| 用 `pkill -f` 时把自己也杀了 | 匹配模式也出现在当前命令行里时会误杀自身。改用具体的 PID |
| 等待别的队列结束时永远在等 | 用 `pgrep -f` 等待时，匹配串不能出现在等待脚本自己的命令行里。改为按 PID 等待(参考 `runs/queue_*.sh` 的写法) |
| 上下文超过 64K 报错 | 虚拟模型名 `stayswitch` 对 LiteLLM 来说是未知模型，所以要通过 `STAYSWITCH_MAX_INPUT_TOKENS` 告诉 terminus-2 真实上限 |
| proxy 启动很慢 | LiteLLM 会去 GitHub 拉价格表。脚本里已经设置 `LITELLM_LOCAL_MODEL_COST_MAP=True` 跳过 |
| 构建卡在下载 uv | 检查 Docker 的 `proxies` 配置和本机代理是否开着;旧的构建步骤可能还卡着，同样的新构建会等它，要先停掉旧的 |

---

## 11. 已知局限

- terminus-2 不是 SWE-bench 主流榜单用的 harness,SWE 上的绝对解决率不能直接和榜单比较，只适合做策略之间的相对比较。
- 本地构建的镜像和官方镜像略有差异(apt 源、未锁版本的包),SWE 用的是 Epoch AI 的 arm64 镜像。
- 每个设置大多只跑了一次，19 个任务的解决率标准误约 11 个百分点。总花费受少数失控的长轨迹影响很大，建议同时看"每千次调用的花费"和"每个任务的调用次数"。
