# 后续研究方向记录（2026-11-02）

基于 `RESULTS.md` 的结论整理。按优先级排序；A 为当前重点，B、C 暂缓。

## A. 演示来源对 BC → RL 微调的影响（当前重点）

问题：用“仿人类”轨迹训练的 BC 做 RL 微调，与用 planner / RL agent 生成的轨迹训练的 BC 做 RL 微调，有什么不同？

- 任务：block-single、block-double（短时程，便于大量 RL 实验）。
- 数据变体（同一脚本控制器，逐个打开“人类特征”）：
  干净（`p_mistake=0`、固定快速度、无噪声，≈ planner）→ 加速度差异 → 加动作噪声 → 加犯错与纠正（默认 OCBench 数据）；
  另加 RL agent 数据（由收敛的 RL 策略 rollout 得到）。需先确认 CPU 控制器 / 采集脚本能调哪些参数。
- BC prior：确定性 BC（L2）与 flow BC。
- 微调方法：
  - 确定性 BC：残差 RL、RLPD（演示放入 buffer）、IBRL。
  - flow BC：残差（EXPO）、噪声空间 steering（DSRL）、基于 Q 的采样选择（Best-of-N 等）。
- 奖励：成功（稀疏）+ 时间惩罚，使微调目标包含演示没有优化的部分（速度、精度）。
- 指标：
  - 初始成功率 J(π0)、达到给定成功率所需交互步数、最终成功率与完成用时。
  - prior 的“可改进空间”：动作熵、π0 样本上的 Q 值离散度 σ_Q、微调后策略相对 π0 的支撑覆盖。
- 核心假设（见对话中的数学分析）：
  - steering 类方法的提升上界由 π0 的支撑与多样性决定。窄的 planner prior 起点高、提升空间小；宽的人类 prior 反之。
  - 残差类方法只能做局部修正。
  - RLPD 的收益取决于数据对最优策略状态分布的覆盖。

## B. 短时程指令条件执行：少量演示 BC → RL（暂缓，A 的推广）

- 子任务环境：
  - 把仿真重置到数据集中任意子任务片段的起点（数据含 `qpos/qvel/dynamics_info`），给一条子目标指令，用控制器 `_subtask_success` 判定成功并记录用时。
  - 各环境 `set_state` 签名不同：基类 `(qpos, qvel)`、hanoi `(qpos, qvel, dynamics_info)`、chamber `(qpos, qvel, dynamics_info, button_states)`、switch `(qpos, qvel, button_states)`。
  - 需先验证重置后观测与数据一致，以及 task_info / 随机布局能被恢复。
- 演示数量：每类子任务 N ∈ {10, 50, 200, 1000, 全部}。另研究演示质量（`speed`、犯错片段）。
- 动机：分层 BC 执行器“慢”是 block/chamber/switch 的主要瓶颈（×4 时间仍大幅上涨），正是 RL 应解决的问题。
- 工程：TPU 节点不能跑 mjwarp（需 CUDA），在线 RL 用 CPU 并行环境（240 vCPU）。ResMLP 4096×8 的 RL 更新速度需实测。

## C. 长时程：VLM 推理器 + 指令执行器（暂缓）

- 把端到端失败拆成感知 / 推理 / 调用时机 / 执行四类误差，用 oracle 逐个替换来隔离。
- 实验 1（无需 VLM）：oracle 高层下的调用方式对比，即每 chunk / 每子任务 / 失败时重规划 / 只在开头规划一次（开环）。
- 实验 2（无需 VLM）：模拟规划延迟（同步等待 vs 异步执行）。
- 实验 3：VLM 2×2 设计，输入（特权状态文字 vs 图像）× 输出（直接 vs 推理链），加上零样本 vs 微调；难度轴为方块数、圆盘数、开关网格大小。
- 前置：执行器接口从特权坐标改为符号化指令；修复 TPU VM 上的渲染（EGL 崩溃）。
- 已知证据：
  - 叠放任务约 40% 的上层超时发生在塔被碰倒之后，说明完全开环很脆弱。
  - hanoi 需要全量数据（1 shard 1.6% → 10 shard 17%）。
