# StaySwitch 使用手册

StaySwitch 是一套研究 coding agent 成本的实验工具。它在 agent 和模型之间放一个代理(proxy),可以：

- 按策略把每一次模型调用路由到不同模型;
- 在发给模型之前压缩历史上下文(自己实现的压缩器，或者串联 CliffCompaction 官方实现);
- 把每次调用的 token 拆成"新输入 / 缓存读 / 缓存写 / 输出"四类记账，并按价目表算钱;
- 回放已有轨迹的前缀，从中间分叉(闭环分叉);
- 按轨迹和总量设花费上限。

任务通过 [Harbor](https://hub.harborframework.com) 运行(Terminal-Bench 2、SWE-bench Verified 等)。支持两种 agent:terminus-2(OpenAI 格式)和 Claude Code(Anthropic 格式,`AGENT=cc`,见第 12 节)。

---

## 1. 整体结构

```
agent:Harbor 里的 terminus-2(OpenAI /chat/completions,带 X-Session-ID)
       或 Claude Code(Anthropic /v1/messages,带 x-claude-code-session-id)
   ▼
[可选] CliffCompaction 官方 proxy   ← 只在 run_cliff.sh 里串联，负责压缩;两种格式都支持
   ▼
StaySwitch proxy (LiteLLM + stayswitch_hook)   ← 路由、压缩、记账、预算、回放
   ▼
模型服务(Tinker / 私有网关 / mock)
```

Claude Code 跑在任务容器里，通过 `host.docker.internal:<端口>` 访问本机的 proxy。

| 路径 | 作用 |
|---|---|
| `src/stayswitch/` | 核心库：计价、会话、策略、信号、压缩、预算、记账 |
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
| **Docker 数据盘写满** | Harbor 每个 trial 都会在任务镜像上构建一层，build cache 会持续增长(一天就到过 72 GB,把 98 GB 的盘写满)。症状很隐蔽:apt 报 "invalid signature"(其实是 gpgv 写不了临时文件)、tmux 装不上、验证器找不到 reward 文件、甚至 agent 的改动静默丢失。检查:`colima ssh -- df -h /var/lib/docker`。清理只用 `docker builder prune -af` 和 `docker container prune -f`,**不要**删 `stayswitch-orig/*` 备份镜像。盘满期间跑过的 trial 要整批重跑 |
| 一台机器同时跑几组实验 | 6 核 12 GB 下合计并发不超过 6 个容器;每组实验用独立的端口对(见 12.6),否则 proxy 会互相顶掉 |

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

扩充任务集(只从已有仓库里挑，按"还没缓存的镜像层字节数"从少到多选，两档难度各半):

```bash
uv run python scripts/swe_images.py extend data/swe-bench-verified --add 21 --max-per-repo 6   # 写 configs/swe_tasks_ext.txt
uv run python scripts/swe_images.py pull data/swe-bench-verified -j 8 --list configs/swe_tasks_ext.txt
```

现在用的是 `configs/swe_tasks_all.txt`:原 19 个 + 扩充 21 个，共 40 个任务。

### 4.3 把 Claude Code 预装进任务镜像

Harbor 的 claude-code agent 启动前会检查容器里有没有 `claude`,有就跳过安装(apt 装 nodejs + 下载二进制，每个 trial 约 5 分钟)。所以把它预装进镜像:

```bash
uv run python scripts/bake_claude.py tools                       # 只需一次：构建 stayswitch-cc-tools(约 5 分钟)
uv run python scripts/bake_claude.py bake configs/swe_tasks_all.txt   # 给每个任务镜像加一层 /root/.local(约 10 秒/个)
uv run python scripts/bake_claude.py bake configs/tb2_tasks_all.txt   # TB2 任务同样适用
uv run python scripts/bake_claude.py restore configs/swe_tasks_all.txt  # 回滚到原镜像
```

- 原镜像备份为 `stayswitch-orig/<task>:latest`,**不要删**;用一次 `docker tag` 原子切换，正在跑的任务看不到中间状态。
- 新增的只有 `/root/.local`(claude 二进制，约 240 MB,当前版本 2.1.286),对 terminus-2 没有行为影响。但为了和旧结果严格可比，**恢复 terminus-2 的旧实验前请先 `restore`**。
- 同一层内容在所有镜像间共享存储。

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
total_usd = 700.0                  # 所有 runs/*/calls.jsonl 合计的上限(不含 mock 模型);2026-10-02 起为 700

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
| `strong_lead` | `weak`, `strong`, `lead_max`, `fail_streak`, `commit` | strong 定位并完成第一次修改，跑完测试后交给 weak;出问题时 strong 回来至少 `commit` 步 |
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
| `cost` | 按 `prices.toml` 算出的花费 |
| `n_messages`, `n_messages_sent`, `ctx_dropped_turns`, `ctx` | 收到的消息数、实际发出的消息数、压缩信息 |
| `input_hash`, `diverged_at` | 分叉时用来检查回放前缀是否一致 |
| `ok`, `error`, `latency_s`, `ts` | 是否成功、错误信息、耗时、时间戳 |

### 7.3 用不同缓存规则重新算钱

```python
from stayswitch.accounting import Call, reprice
from stayswitch.pricing import PriceTable
bill = reprice(calls, PriceTable.load("configs/prices.toml"), ttl_s=None)   # ttl_s=None 表示持久缓存
bill.cache_aware, bill.cache_oblivious, bill.switches, bill.cold_calls
```

### 7.4 当前花费

```bash
uv run python -c "from stayswitch.budget import spent_in_logs; print(spent_in_logs('runs/*/calls.jsonl'))"
```

注意：总预算是全局台账，一旦超过上限，所有正在跑的轨迹都会被 proxy 用 `budget_stop` 截停，正在跑的那组实验就作废了。排大批实验前先估算花费;`report_arms.py` 会把被"总预算"截停的轨迹剔除并在 stderr 报警(单轨迹 $3 的截停照常计入，各组一致)。

### 7.5 压缩实验的分析脚本

| 脚本 | 用途 |
|---|---|
| `scripts/perfect_cache_cost.py <run>...` | 按 CliffCompaction 论文附录 A.1 的"完美缓存"规则(只看 prompt 长度序列)重算花费，同时给真实花费、命中率、峰值上下文、压缩次数 |
| `scripts/report_arms.py <基线run> <run>...` | 单种子：每组的解决率、调用数、每任务和每调用花费;与基线逐任务配对的差值 ± 标准误;冗余步数 e 的估计;逐任务对错表 |
| `scripts/report_seeds.py name=run,run_r2,run_r3 ...` | 多种子：每组各种子的均值 ± 标准差、合并解决率 ± 二项标准误、按 (任务, 种子) 配对的差值。`--json` 输出 |
| `scripts/pass_at_k.py name=run,run_r2,run_r3 ...` | 把各种子当作 k 次 rollout:pass@1、oracle pass@k、k 次 rollout 的花费、每美元解决数 |
| `scripts/price_regimes.py name=run,... ...` | 不重跑，按四种缓存价目(Tinker、Anthropic 式、DeepSeek 式、OpenAI 式)重算每组花费，并给出 EOQ 在该价目下推出的触发点 |
| `scripts/action_mix.py name=run,... ...` | 机制分析：每任务的探索 / 编辑 / 测试步数、重读文件次数、循环比例 |

trial 与 Claude Code 会话的对应：会话 ID 就是 `jobs/<run>/*/<trial>/agent/sessions/projects/*/<session_id>.jsonl` 的文件名。同一任务多次 trial 时只取最新的。

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
| 停掉实验后，还有 trial 在不断启动 | `uv run harbor ...` 有两层进程：外层是 `uv`,里面才是真正的 harbor(Python)。只结束外层，里面的进程会变成孤儿继续跑。要把两层都按 PID 结束(`ps -eo pid,ppid,command \| grep "harbor run"`),再删除对应的容器 |
| 用 `pkill -f` 时把自己也杀了 | 匹配模式也出现在当前命令行里时会误杀自身。改用具体的 PID |
| 等待别的队列结束时永远在等 | 用 `pgrep -f` 等待时，匹配串不能出现在等待脚本自己的命令行里。改为按 PID 等待(参考 `runs/queue_*.sh` 的写法) |
| 上下文超过 64K 报错 | 虚拟模型名 `stayswitch` 对 LiteLLM 来说是未知模型，所以要通过 `STAYSWITCH_MAX_INPUT_TOKENS` 告诉 terminus-2 真实上限 |
| proxy 启动很慢 | LiteLLM 会去 GitHub 拉价格表。脚本里已经设置 `LITELLM_LOCAL_MODEL_COST_MAP=True` 跳过 |
| 构建卡在下载 uv | 检查 Docker 的 `proxies` 配置和本机代理是否开着;旧的构建步骤可能还卡着，同样的新构建会等它，要先停掉旧的 |
| Claude Code 一直没做 auto-compact,上下文顶到 64K 后 Tinker 报 `PromptTooLongException` | 环境变量名写错了。二进制只认 `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`,`CLAUDE_CODE_AUTOCOMPACT_PCT_OVERRIDE` 会被忽略。关闭原生压缩用 `DISABLE_AUTO_COMPACT=1` |
| Claude Code 报 `reasoning_effort is not supported` | Claude Code 会带 `output_config` / `thinking`,LiteLLM 转成 OpenAI 格式时变成 `reasoning_effort`,Tinker 拒收。hook 已对 `/v1/messages` 自动去掉这些字段 |
| Claude Code 报 "Write was called with input that could not be parsed as JSON" | Tinker 把多个并行工具调用放在一个流式 chunk 里，LiteLLM 会把参数拼成非法 JSON。`stayswitch/litellm_patches.py` 已在 hook 启动时修补 |
| 用 zsh 把多个 run id 放进变量传给脚本，结果脚本只收到一个参数 | zsh 默认不对 `$VAR` 分词。用 `${=VAR}` 或直接写出参数 |
| 构建 TB2 镜像报 "build tag cannot contain a digest" | 该任务的 `docker_image` 用 `@sha256` 固定，无法本地构建，换任务 |

---

## 11. 已知局限

- terminus-2 不是 SWE-bench 主流榜单用的 harness,SWE 上的绝对解决率不能直接和榜单比较，只适合做策略之间的相对比较。
- 本地构建的镜像和官方镜像略有差异(apt 源、未锁版本的包),SWE 用的是 Epoch AI 的 arm64 镜像。
- 每个设置大多只跑了一次，19 个任务的解决率标准误约 11 个百分点。总花费受少数失控的长轨迹影响很大，建议同时看"每千次调用的花费"和"每个任务的调用次数"。

---

## 12. Claude Code harness 与 API 接入

### 12.1 链路

```
Claude Code 2.1.286(任务容器内,Harbor 的 claude-code agent;子类 stayswitch.agents:StaySwitchClaudeCode)
   │  POST /v1/messages(Anthropic 格式，流式),请求头 x-claude-code-session-id
   │  ANTHROPIC_BASE_URL=http://host.docker.internal:<端口>,模型名固定为 "stayswitch"
   ▼
[可选] CliffCompaction 官方 proxy(--anthropic-upstream 指向下一跳)
   ▼
StaySwitch proxy(LiteLLM;/v1/messages 走它的 Anthropic→OpenAI 适配器)
   ▼
Tinker 的 OpenAI 兼容接口(Qwen3.6-35B-A3B 等)
```

本机直接跑(不进容器)也可以，用于冒烟测试:

```bash
scripts/start_proxy.sh configs/cc_smoke.toml 4000 &
env -u BUN_OPTIONS CLAUDE_CONFIG_DIR=$(mktemp -d) ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_API_KEY=sk-dummy \
  ANTHROPIC_MODEL=stayswitch ANTHROPIC_DEFAULT_HAIKU_MODEL=stayswitch ANTHROPIC_DEFAULT_SONNET_MODEL=stayswitch \
  ANTHROPIC_DEFAULT_OPUS_MODEL=stayswitch CLAUDE_CODE_SUBAGENT_MODEL=stayswitch \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude -p "..." --dangerously-skip-permissions --output-format json
```

- `CLAUDE_CONFIG_DIR` 用一个空目录，避免读到本机的 MCP、插件和记忆，导致 prompt 变大、结果不可复现。
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` 后，每个 agent 步只有一次请求，没有后台的 Haiku 调用。
- 所有档位(Haiku/Sonnet/Opus/子 agent)都指到 `stayswitch`,由 proxy 决定实际模型。

### 12.2 proxy 对 `/v1/messages` 做的事(`proxy/stayswitch_hook.py`)

| 处理 | 原因 |
|---|---|
| 会话 ID 取 `x-claude-code-session-id`(也认 `X-Session-ID`) | Claude Code 自带，不用注入 |
| 去掉 `output_config` / `thinking` / `context_management` | 否则会被转成 Tinker 不支持的 `reasoning_effort` |
| 子 agent 单独成轨迹:`<sid>~sub-<hash>` | 子 agent 和主 agent 共用会话头，但 system prompt 和工具集不同(按 `agent_key` 区分);模型和预算跟随主轨迹 |
| 同一会话收到相同输入的请求视为重试，不增加步数 | Claude Code 在流式失败时会改用非流式重发 |
| 记录工具调用、thinking 和 usage | 用于回放分叉，回放时 prompt 能逐字节一致 |
| 回放由 `stayswitch/replay_server.py` 提供(端口 4390,hook 按需启动) | LiteLLM 的 mock 只支持文本、不支持流式和 `tool_use` |
| 修补 LiteLLM 的流式翻译(`stayswitch/litellm_patches.py`) | 并行工具调用的参数会被拼成非法 JSON |
| token 估计计入 system + 工具定义 | Claude Code 每次请求固定带约 18K token 的 system 和 26 个工具定义 |

`signals.py`(失败、循环检测)和 `context.py`(压缩)都能读 Anthropic 的 `tool_use` / `tool_result` 块。

### 12.3 在 Harbor 里跑

```bash
AGENT=cc scripts/run_model.sh swe-bench/swe-bench-verified configs/swe_tasks_all.txt configs/cc_mid.toml 4030 3
AGENT=cc DISABLE_AUTO_COMPACT=1 scripts/run_cliff.sh configs/swe_tasks_all.txt configs/cc_mid_cliff45.toml fixed 45000
AGENT=cc DISABLE_AUTO_COMPACT=1 EOQ_VERSION=3 EOQ_CEILING=46000 \
  scripts/run_cliff.sh configs/swe_tasks_all.txt configs/cc_mid_eoq3.toml eoq 2.3
```

`scripts/run_harbor_cc.sh` 的环境变量:

| 变量 | 默认值 | 作用 |
|---|---|---|
| `CC_BASE_URL` | `http://host.docker.internal:$STAYSWITCH_PORT` | 容器里 Claude Code 访问的地址(串联 Cliff 时由 `run_cliff.sh` 设为 Cliff 端口) |
| `CC_COMPACT_PCT` | 25 | 原生 auto-compact 的触发比例(Claude Code 假设窗口 200K,25% 约 50K) |
| `DISABLE_AUTO_COMPACT` | 不设 | 设为 1 时关闭原生压缩。**所有 Cliff / EOQ 组都必须设**,否则两个压缩器会互相干扰 |
| `STAYSWITCH_CC_AGENT` | `stayswitch.agents:StaySwitchClaudeCode` | 使用的 agent 类 |

### 12.4 CliffCompaction 和 EOQ 的参数

| 变量 | 作用 |
|---|---|
| `CLIFF_THOUGHT_MAX_CHARS=300`、`CLIFF_THINKING_MAX_CHARS=300` | 和论文一致：被压缩区域里每轮 assistant 文本和 thinking 截到 300 字符(官方默认不限) |
| `KEEP_RECENT` | 保留最近几轮原文(默认 3) |
| `EOQ_VERSION` | 1:全历史平均增长 + 猜测的 L0;2:周期内增长 + 实测 L0(触发偏早，已弃用);3:全历史增长 + 实测 L0(当前推荐) |
| `EOQ_CEILING` | 阈值上限(Cliff 的估计 token)。Cliff 按字符数 / 4 估计，比真实 token 低约 30%,64K 窗口下设 46000 |
| `EOQ_FAILURE_TRIGGER=1` | v3f:连续 3 次失败或重复动作时提前压缩，只保留最后一轮(两次之间至少隔 4 步) |

冗余步数参数(`eoq` 模式的最后一个参数)目前用 2.3,来自配对运行的实测。

### 12.5 运行速度

- 预装镜像后，一个 SWE trial 约 5 到 8 分钟(agent 执行 1 到 5 分钟，验证约 3 分钟)。
- terminus-2 每个 trial 要装 tmux,约 6 到 7 分钟。
- TB2 轨迹很长：一个任务约 400 次调用、$1.6、45 分钟。

### 12.6 端口分配(多组实验并行时)

| 端口 | 用途 |
|---|---|
| 4000 / 4005 / 4011–4019 | 本机冒烟、mock 测试 |
| 4001 + 8257 | terminus-2 实验(router + Cliff) |
| 4020 / 4021 + 8259 | 另一个会话的实验 |
| 4030 + 8330、4031 + 8331、4032 + 8332、4033 + 8333 | Claude Code 实验的四条并行队列 |
| 4390 | 回放服务(多个 proxy 共用，无状态) |
| 8765 | GitHub release 缓存 |

停止进程只按端口或 PID(`lsof -tiTCP:<端口> -sTCP:LISTEN \| xargs kill`),**不要**用 `pkill -f "litellm --config"`,它会杀掉所有会话的 proxy。

---

## 13. 当前实验方案(2026-10-02)

研究问题：压缩时机能否从价目表解析地推出(EOQ 规则 L\* = L₀ + √(2gC/c_r)),不调参就在不同 harness 上落到各自的最优区间。设置参照 CliffCompaction 论文(arXiv 2609.26779,对照见 `docs/CLIFF_ALIGNMENT.md`),结果见 `docs/RESULTS_CC.md`。

公共设置:mid 模型(Qwen3.6-35B-A3B,Tinker,64K 窗口),SWE-bench Verified 40 题(`configs/swe_tasks_all.txt`),每组 3 个种子;成本按论文的完美缓存规则计。

| 批次 | run id | 组 | 状态 |
|---|---|---|---|
| Claude Code 主矩阵 | `cc_mid{,_r2,_r3}`、`cc_mid_cliff{45,40,32}*`、`cc_mid_eoq{1,2}*` | 原生 auto-compact、Cliff 45K/40K/32K、EOQ v1/v2 | 完成;盘满窗口内的 trial 在重跑(`runs/rerun_diskwindow.sh`) |
| EOQ v3 / v3f(Claude Code) | `cc_mid_eoq3*`、`cc_mid_eoq3f*` | 修正后的 EOQ;失败触发提前压缩 | 进行中(`runs/queue_cc_eoq3.sh` → `queue_cc_eoq3f.sh`) |
| 跨 harness 迁移 | `tm_mid_{native,cliff16,cliff32,eoq3}*` | terminus-2:原生摘要、Cliff 16K/32K、EOQ v3 | 进行中(`runs/queue_tm.sh`) |
| 跨模型 | `cc_strong_{native,cliff45,eoq3}` | 397B,1 个种子 | 排队(`runs/queue_cc_strong.sh`) |
| 长时程 | `cc_tb2_{native,cliff45,eoq3}{,_r2}` | Terminal-Bench 2 的 12 题,2 个种子 | 排队(`runs/queue_cc_tb2.sh`) |

队列脚本都在 `runs/` 下，按 PID 串联(后一个等前一个的 PID 退出)。查看进度:`grep "start \|all done" runs/queue_*.log`。

出表:

```bash
A="native=cc_mid,cc_mid_r2,cc_mid_r3 cliff45=cc_mid_cliff45,cc_mid_cliff45_r2,cc_mid_cliff45_r3 eoq1=cc_mid_eoq1,cc_mid_eoq1_r2,cc_mid_eoq1_r3"
uv run python scripts/report_seeds.py ${=A}      # zsh 下用 ${=A};bash 下直接 $A
uv run python scripts/pass_at_k.py ${=A}
uv run python scripts/price_regimes.py ${=A}
uv run python scripts/action_mix.py ${=A}
```

主图:`docs/fig_dose_response.html`(解决率和每任务花费对压缩触发点)。
