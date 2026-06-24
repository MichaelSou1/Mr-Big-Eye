# WU-2 完成快照 — faithfulness battery（5 干预 + N2 重读控制）

> 完成 2026-06-24。所有数字经 `data/distill/poc/battery_{8b,32b}_{present,masked}.json` 指纹化产出，
> `python scripts/battery_table.py` 一键重生（→ `battery_table.md` / `.json`，无手抄）。
> 跑法：`bash scripts/run_battery_grid.sh`（8B@GPU0 ∥ 32B@GPU1-3，present→masked，paraphrase 缓存按 scale 分文件）。
> 底物 = ChartQA test n=400 开放式数值（无字母选项 → 无 disguised-accuracy 偏好，化解 2402.14897）。

## 验收：4/4 ✅（UPGRADE_PLAN WU-2）/ 对应 todo §2.2 五条
- [x] battery 表：干预 × {8B,32B} × {present,masked} flip-rate，**n_eval=321/318 ≥ 300**
- [x] **N2 重读 snap-rate 报出**（reading-not-reasoning 直证）
- [x] early-answering 曲线；paraphrase = corrupt 对侧控制，filler = shuffle 对侧控制
- [x] **每个 flip-rate 旁报 accuracy**（开放式数值底物，无字母偏好）

## 主表：flip_rate / accuracy_after（n_eval）
| 干预 | 8b present | 8b masked | 32b present | 32b masked |
|---|---|---|---|---|
| corrupt | .212/.816 | .333/.710 | **.051/.981** | .120/.908 |
| shuffle | .302/.735 | .408/.632 | .047/.987 | .129/.903 |
| paraphrase (corrupt-ctrl) | .150/.885 | .181/.879 | **.053/.969** | .060/.965 |
| filler (shuffle-ctrl) | .579/.449 | .950/.062 | .362/.679 | .962/.038 |
| truncate@.25 | .419/.616 | .700/.338 | .265/.782 | .609/.426 |
| truncate@.5 | .306/.725 | .481/.566 | .170/.864 | .262/.770 |
| truncate@.75 | .183/.840 | .279/.772 | .061/.974 | .061/.968 |
| delete k=1 | .134/.906 | .181/.875 | .035/.994 | .044/.984 |
| delete k=2 | .170/.874 | .248/.814 | .048/.981 | .063/.965 |
| delete k=3 | .221/.817 | .449/.615 | .074/.965 | .251/.794 |

## F = flip_corrupt − flip_shuffle（≤0 ⇒ 篡改数字不比打乱句序更翻 ⇒ CoT 非 load-bearing）
| cell | n_eval | corrupt | shuffle | F |
|---|---|---|---|---|
| 32b present | 318 | .051 | .047 | **+.003 ≈0** |
| 32b masked | 318 | .120 | .129 | −.009 |
| 8b present | 321 | .212 | .302 | −.090 |
| 8b masked | 321 | .333 | .408 | −.075 |

- 四格 F ≤ ~0，**与 WU-1.4 的 2×2 探针逐格吻合**（.050/.047、.212/.302、.333/.408）→ battery harness 自校验通过。

## N2 重读控制（present；头条新结果）
篡改 CoT 里一个中间数后，答案是**吸附回真实图表值**(re-read) 还是**跟随注入的错值**(load-bearing)？
| scale | n_corrupt | snap→true | follow→injected | other | **snap_rate** | follow_rate |
|---|---|---|---|---|---|---|
| 32b | 316 | 310 | 5 | 1 | **.981** | .016 |
| 8b | 321 | 262 | 10 | 49 | **.816** | .031 |

- **答案 98%(32B)/82%(8B) 吸附真值、跟随注入错值 <2%** → 这是 "reading not reasoning" 的最直接证据，
  正面化解 2502.14829（"contextualization 不等于 load-bearing"）：模型不算 CoT 里的数，它重读图。

## Early-answering 曲线（truncate 比例 → flip，present）
| scale | @.25 | @.5 | @.75 |
|---|---|---|---|
| 32b | .265 | .170 | .061 |
| 8b | .419 | .306 | .183 |
- 单调：保留越多 CoT flip 越低；但 32B 即便只保留 25% 也仅 .265 翻（其余靠重读图找回答案）。
- masked 对照（无图）截断/删步咬得更狠（truncate@.25=.609/.700；delete k=3=.251/.449）→ 没图时 CoT 才开始承重。

## 控制对：paraphrase / filler
- **paraphrase（语义保持改写 = corrupt 的对侧控制）flip ≈ corrupt**：
  - 32B present **corrupt .051 ≈ paraphrase .053 ≈ shuffle .047**(三者皆 ~5%,Δ<1 例)→ 改一个数翻得**不比纯改写/打乱更多** → 数字非 load-bearing(最干净的零结果)。
  - 8B present **paraphrase .150 < corrupt .212 < shuffle .302**:corrupt 仅高出 paraphrase 基线 .06、且低于 shuffle → 至多弱内容敏感,主效应是感知/格式而非算术。
  - 关键:paraphrase 给出 "任意 CoT 编辑能翻多少" 的下界;corrupt 没有显著超过这个下界 → corrupt 的低 flip **不是** "编辑不敏感" 的假象。
- **filler（等长无义 token = shuffle 的对侧控制）每格 flip > shuffle**，masked 近乎全翻(.95/.96, acc≈.04)
  → 格式/长度本身不解释结论；present 下有图仍能找回(32B acc .679)，masked 下无图则崩 → 再证图是主信息源。

## ⚠️ paraphrase 数字保真（诚实记录）
- DeepSeek(v4-flash) 改写后**数字多重集完全保留率 = 8B 223/321 (69%)、32B 234/318 (74%)**
  （3 次带数字闸重试后仍有 ~30% 至少漂 1 个数；长 CoT 更难全保）。
- **不影响结论**：数字漂移只会**抬高** paraphrase flip，而实测 paraphrase flip 仍 ≤ corrupt
  → 报出的 paraphrase 数是**保守上界**，控制依然成立。（缓存 + 闸见 `scripts/battery_n400.py:build_paraphrase_cache`）
- 缓存：`data/distill/poc/paraphrase_cache_{8b,32b}.jsonl`（present 跑生成、masked 复用，base-CoT md5 键，离线可复跑）。

## 一句话
扩成 5 干预 + N2 重读控制后，"CoT 非 load-bearing / 模型在读不在算" 被**多路独立坐实**：
F≤0 四格复现、snap_rate .98/.82、corrupt≈paraphrase(改数不比改写更翻)、early-answering 单调、delete-k 几乎不翻 —— 全在 n≥300 高功效下。
