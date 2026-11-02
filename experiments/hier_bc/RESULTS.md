# 分层 BC（oracle 高层 + 子目标条件低层 BC）实验记录

2026-10-31 ~ 2026-11-02，TPU v4-8 ×3（tpu_node_4/5/6）。

## 方法

- **高层**：OCBench 的脚本控制器（`ocbench/oracles/controllers`），`p_mistake=0`，读特权状态，每次给出一个子目标。
- **低层**：确定性 BC（`agents/bc.py`，L2，ResMLP [4096]×8，chunk 25，lr 1e-4，batch 1024），输入 = 观测 + 子目标编码。
  训练数据 = 元数据里所有 `subtask_success` 的片段（包括失败 episode 和高层 mistake 中的成功片段）。
- **子目标完成判定**：子任务成功（控制器的 `_subtask_success`）+ 夹爪接触 < 0.5（switch 除外）+ 末端从松手点移开 ≥ 0.1m。
  超时（block/chamber 1250 步，hanoi 1875，switch 600）后高层重新选子目标。
- **平坦 BC 基线**：同一脚本 `--flat=1`，只用成功 episode、无子目标（论文的 BC 设置）。
- **评估**：500 episodes，episode 时间上限 ×1（benchmark 原值）/ ×2 / ×4，用于判断失败是否来自“时间不够”。
  ×2/×4 只是诊断用，与 benchmark 数字不可直接比较。

代码：`impls/hier_bc.py`（训练 + 评估）、`impls/hier_common.py`（子目标编码、并行评估）、`impls/eval_subgoal_bc.py`（block 旧 ckpt 评估）。

```bash
# 分层 BC，1 shard
python hier_bc.py --env_name=chamber-hard-task1-v0 --dataset_root=/mnt/ram/ocb --num_shards=1 --delete_shards=1 \
  --offline_steps=1000000 --final_eval_horizons=1,2,4
# 平坦 BC
python hier_bc.py --flat=1 ...
# 全量数据（逐 shard 下载→过滤→删除，hanoi-triple 约 167G 内存）
python hier_bc.py --num_shards=0 --delete_shards=1 --offline_steps=2000000 ...
```

## 1. 主结果：平坦 BC vs 分层 BC（1 shard，1M steps）

成功率 %，×1 / ×2 / ×4 时间上限。论文列为论文 Table 6 的 BC（全量数据、2M steps，不可直接比较）。

| 任务 | 平坦 BC | 分层 BC | 论文 BC |
|---|---|---|---|
| block-quadruple-task1 | 进行中 | 36.4 / 78.0 / 89.0 ¹ | 0 |
| block-quadruple-task2 | 0.0 / 0.0 / 0.0 | 10.2 / 23.2 / 32.8 | 3 |
| block-quadruple-task3 | 0.0 / 0.2 / 0.2 | 21.2 / 49.0 / 64.0 | 2 |
| block-triple-task1 | 0.2 / 1.0 / 4.6 | 63.0 / 90.2 / 95.2 | 1 |
| block-triple-task2 | 0.6 / 0.8 / 3.6 | 41.8 / 65.2 / 74.4 | 27 |
| chamber-medium-task1 | 6.6 / 29.6 / 51.4 | 67.6 / 83.0 / 87.2 | 7 |
| chamber-medium-task2 | 4.2 / 17.6 / 27.0 | 56.8 / 76.0 / 84.4 | 10 |
| chamber-hard-task1 | 0.2 / 4.8 / 16.0 | 33.2 / 64.2 / 73.6 | 0 |
| chamber-hard-task2 | 0.0 / 0.0 / 1.6 | 18.0 / 33.8 / 41.6 | 1 |
| switch-5x5-task1 | 0.0 / 0.0 / 0.0 | 43.2 / 83.0 / 96.8 | 0 |
| switch-5x5-task2 | — | 46.6 / 69.8 / 83.6 | 0 |
| hanoi-triple-task1 | 2.8 / 4.2 / 4.4 | 1.6 / 1.6 / 1.6 | 18 |
| hanoi-triple-task1，**全量 10 shard，2M** | — | **17.0 / 21.4 / 21.8** | 18 |
| hanoi-triple-task2 | 1.2 / 2.2 / 2.2 | 1.2 / 2.0 / 2.0 | 14 |

¹ 原 `subgoal_bc.py` 的 500k ckpt（完成判定为“末端距方块 ≥0.1m”）；用 `hier_bc` 的评估器重评为 34.0±2.1%。

结论：
- 平坦 BC 在原时间上限下基本为 0（chamber-medium 4–7%，与论文接近）。分层 BC 在 block/chamber/switch 上高 20–60 个点。
- **速度是 block/chamber/switch 的主要瓶颈**：×2 平均 +20 个点，×4 继续上升。平坦 BC 也有同样现象（chamber-medium-task1 7%→51%）。
- **hanoi 的瓶颈是数据量**：1 shard 时 1.6%，全量 17%（与论文平坦 BC 18% 持平）。加时间帮助很小（卡死而不是慢）。

## 2. 子任务级诊断

- **chamber**：button/drawer/window 完成率 86–100%（250–375 步），cube 只有 30–67%（750–800 步）→ cube 是瓶颈。
- **switch**：子目标超时时 30–40% 按钮状态其实已正确 → 完成判定（末端移开 0.1m）对按钮过严。
  训练越久越差：100k 时 100-episode eval 94%（task2 58%），1M 时 43–47%。
- **hanoi（1 shard）**：77% 的失败是策略悬停在源柱圆盘上方约 0.3m 不下抓（多为分散局面下搬大盘）；21% 为非法叠放/落地导致环境终止。
  全量后每 episode 完成子目标 1.65→3.5，子目标完成率 45%→73%。
- **block-quadruple 三个任务差异**（200 eps，×4，`level_diag.py`）：task1 = 4 块分别放地面，task2 = 叠 4 层塔，task3 = 两座紧贴的 2 层塔。

  | | 成功率 | 第 1 层完成率 | 第 2 层 | 第 3 层 |
  |---|---|---|---|---|
  | task1 | 89.5% | 38% | — | — |
  | task2 | 30.0% | 18% | 15% | 20% |
  | task3 | 67.0% | 45% | 25% | — |

  上层子目标超时时约 40% 下层方块已不在（塔被碰倒），高层仍坚持当前子目标直到超时 → 碰倒已完成进度是叠放任务的主要失败模式。
  演示数据本身：episode 成功率 task1 64% / task2 61% / task3 33%。

## 3. block-quadruple-task1 瓶颈分析（`subgoal_bc` 500k ckpt，500 eps）

| 设置 | 成功率 |
|---|---|
| 原始（T=5000） | 36.4 ± 2.2 |
| T=10000 | 78.0 ± 1.9 |
| T=20000 | 89.0 ± 1.4 |
| T=20000，无子目标超时 | 89.4 ± 1.4 |
| 子目标超时 800 / 2500 | 35.0 / 38.8 |
| 去掉末端距离条件 / 只看目标 / hold 判定 | 14.8 / 14.2 / 22.6 |

- 策略慢而不是卡死：无超时时子目标中位 1200 步、p90 3075 步（数据中约 1000）。超时基本是空操作（91% 重新下达同一方块）。
- 放宽完成判定反而更差：过早切换会碰掉已放好的方块。

## 文件

- `results/hier_bc_1shard/<env>/`、`results/flat_bc_1shard/<env>/`、`results/hier_bc_full/<env>/`：`eval.csv`（训练中 100-episode eval）与 `final_eval_*.json`（×1/×2/×4 详细指标）。
- `results/subgoal_bc_block_quadruple_task1/`：第 3 节的各设置。
- `summarize_hier.py <results/xxx>`：打印曲线与子任务指标；`hanoi_diag.py`、`level_diag.py`：基于评估 episode pickle（带 `diag` 结束状态快照）的诊断。
