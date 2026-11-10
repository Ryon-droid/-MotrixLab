# 第四周作业

- **[提交文档：第四周作业.pdf](第四周作业.pdf)**（含按题号作答、截图和视频链接）
- [可编辑答案](answer.md)
- [8段完整回合 MP4](videos/)
- [原始要求](（第四周）强化学习与仿真软件第五期实训营作业.pdf)

完成内容：CartPole、Pendulum、Acrobot 的实际 PPO 训练和完整回合回放；Pendulum 默认能量奖励 A 组与仅去掉能量贡献的 B 组；dm-reacher 场景、源码分析及八要素选做题。

## 本次真实结果

| 实验 | 批量步（每环境步） | 并行环境数 | 环境转移数 | 回放 |
| --- | --- | --- | --- | --- |
| CartPole | 4096 | 128 | 524288 | seed 0，1000步/10 s，正常时间截断 |
| Pendulum A | 8192 | 128 | 1048576 | seeds 0/1/2，各400步/10 s，末帧偏角均约6.38° |
| Pendulum B | 8192 | 128 | 1048576 | seeds 0/1/2，各400步/10 s，末帧偏角均约2.83° |
| Acrobot | 8192 | 128 | 1048576 | seed 0，500步/10 s，超时两个结束标志均为True |

A/B 都从头训练，seed=42、配置一致，唯一任务差异是总奖励中的 `energy_reward` 变为 `0.0 * energy_reward`。两组6个回合都曾进入15°直立范围，末帧也在范围内。有限种子与训练预算的局限、能量公式与XML几何的差异，在PDF第二题中分析；不能根据两组总奖励大小判断优劣。

## 安装与运行

所有命令在 `week4/` 目录执行，Python要求3.10。源码已随作业提供，**不依赖第二、三周目录，也不依赖完整官方仓库**。Linux示例：

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

本机 `.venv` 已准备好，可以直接使用。为减少重复下载，本机通过 `.pth` 复用已安装的同版本 Torch；新机器按上述 requirements 安装即可。Torch wheel带CUDA，本次用 `CUDA_VISIBLE_DEVICES=''` 明确选择CPU。渲染需要图形/Vulkan驱动；训练与轨迹评估无需窗口。构建PDF/截取TensorBoard时使用系统Chrome：`/usr/bin/google-chrome`。

Windows可用 `uv venv --python 3.10 .venv`，将以下 `.venv/bin/python` 替换成 `.venv\Scripts\python.exe`；设置 `$env:CUDA_VISIBLE_DEVICES=""` 后运行Python命令。Python本身的训练/评估入口不依赖Linux路径，截图与排版脚本使用本次Linux字体和Chrome路径。

### 加载已提交模型回放一个完整回合

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python code/evaluate.py --env cartpole --seeds 0 --policy runs/cartpole/final_agent.pt
CUDA_VISIBLE_DEVICES='' .venv/bin/python code/evaluate.py --env pendulum --group A --seeds 0 1 2 --policy runs/pendulum_A/final_agent.pt
CUDA_VISIBLE_DEVICES='' .venv/bin/python code/evaluate.py --env pendulum --group B --seeds 0 1 2 --policy runs/pendulum_B/final_agent.pt
CUDA_VISIBLE_DEVICES='' .venv/bin/python code/evaluate.py --env acrobot --seeds 0 --policy runs/acrobot/final_agent.pt
```

入口采用官方play的均值动作，并捕获自动重置**之前**的末帧。每组的NPZ记录每个控制步的qpos、qvel、obs、动作、奖励；`episodes.json`记录结束标志、首次直立时刻、模型SHA-256。

将真实轨迹渲染为MP4，或直接打开已提交的 `videos/*.mp4`：

```bash
.venv/bin/python code/render_replays.py --task cartpole
.venv/bin/python code/render_replays.py --task pendulum_A
.venv/bin/python code/render_replays.py --task pendulum_B
.venv/bin/python code/render_replays.py --task acrobot
```

相机参数、帧号、帧率见 `evidence/replay/*/capture_manifest.json`。相同轨迹在官方引擎中重新设置状态并渲染，不修改模型参数、不补算控制动作；视频从初始状态到真实结束状态，保留最后一帧的显示时间。

### 重新训练

提交时使用的精确命令见 `answer.md` 和 `runs/*/manifest.json command`。默认实验目录已有结果时，入口拒绝覆盖。重做实验请使用独立目录，例如：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python code/train.py --env cartpole --num-envs 128 --steps 4096 --seed 42 --output reruns/cartpole
CUDA_VISIBLE_DEVICES='' .venv/bin/python code/train.py --env pendulum --group A --num-envs 128 --steps 8192 --seed 42 --output reruns/pendulum_A
CUDA_VISIBLE_DEVICES='' .venv/bin/python code/train.py --env pendulum --group B --num-envs 128 --steps 8192 --seed 42 --output reruns/pendulum_B
CUDA_VISIBLE_DEVICES='' .venv/bin/python code/train.py --env acrobot --num-envs 128 --steps 8192 --seed 42 --output reruns/acrobot
```

训练使用v0.3.0原始网络/PPO构造器和SKRL训练器，仅更改并行数、批量步数和日志设置。运行入口在构造配置后设置字段，避免v0.3.0 `cfg_override` 重构dataclass时的默认值覆盖问题。`final_agent.pt`为最后更新后的权重；回放统一使用final，未按结果选择最佳checkpoint。

### dm-reacher与核验

```bash
.venv/bin/python code/view.py --env dm-reacher --seed 0
.venv/bin/python code/render_replays.py --task reacher
CUDA_VISIBLE_DEVICES='' .venv/bin/python code/verify.py
```

`view.py`调用官方NpRenderer；`scene_annotated.png`标注两连杆、两关节、末端和目标。`desktop_viewer.png`为真实桌面截图中裁出的MotrixRender窗口，去掉了窗口外的IDE界面。`verify.py`检查源码哈希、A/B唯一修改、实际配置一致性、同状态奖励差、超时标志、单摆几何和reacher的实际10维观测/目标采样/奖励边界。

### TensorBoard与PDF

```bash
.venv/bin/tensorboard --logdir runs --host 127.0.0.1 --port 6006
.venv/bin/python code/capture_evidence.py
.venv/bin/python code/build_report.py
.venv/bin/python code/check_artifacts.py
```

`capture_evidence.py`使用6007端口依次打开各任务的真实TensorBoard并截屏，同时从原始文件生成代码/日志浏览器截图。默认平滑0.6；原始标量见CSV。HTML与PDF由同一份 `answer.md` 构建，图片直接嵌入PDF，视频以可访问的GitHub链接引用并随目录提交MP4。

## 文件与版本

- `runs/`：每项实验的配置、完成清单、最终模型、定期/最佳checkpoint、TensorBoard事件。
- `evidence/`：训练/回放/渲染原始日志、真实TensorBoard截图、源代码截图、状态轨迹和核验结果。
- `videos/`：三任务与A/B实验共8段完整回合，不使用官方示例视频。
- `source/`：官方v0.3.0最小未修改源码和真实纹理，附LICENSE/NOTICE。
- `code/pendulum_no_energy.py`：独立B组文件，仅第88行修改能量项贡献。
- `code/prepare_sources.py`：从指定官方提交归档提取快照的脚本；常规运行不需要再提取。

官方提交：[`e1421d10559c825e72bab0f3cbb517c333d67e16`](https://github.com/Motphys/MotrixLab/tree/e1421d10559c825e72bab0f3cbb517c333d67e16)。全部37个快照文件校验值在 `evidence/source_manifest.json`；地面纹理验证为同一Git LFS对象。所有作答均以该版本实现为准。

## 视频快速入口

- [CartPole seed0](videos/cartpole_seed0.mp4)、[Acrobot seed0](videos/acrobot_seed0.mp4)
- Pendulum A：[seed0](videos/pendulum_A_seed0.mp4)、[seed1](videos/pendulum_A_seed1.mp4)、[seed2](videos/pendulum_A_seed2.mp4)
- Pendulum B：[seed0](videos/pendulum_B_seed0.mp4)、[seed1](videos/pendulum_B_seed1.mp4)、[seed2](videos/pendulum_B_seed2.mp4)
