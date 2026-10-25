# Week 2：第 5–7 讲课后作业

完整作答请看 **[answer.md](answer.md)**：30 道单选题及简析、第 5 讲字段归类和改错、第 6 讲 simple_walker 接入代码、第 7 讲 CartPole 实操证据。

## 文件说明

| 路径 | 内容 |
| --- | --- |
| `answer.md` | 全部题目作答与实际实验结论。 |
| `code/` | simple_walker 配置和实现、接入脚本、CartPole 训练和评估脚本。 |
| `evidence/` | 原始命令日志、截图、曲线、CSV、环境版本和校验清单。 |
| `runs/cartpole/` | 实际 CartPole TensorBoard events 和全部 checkpoint。 |
| `runs/simple_walker/` | 实际 simple_walker 训练 events 和 checkpoint。 |

## 复现环境

使用官方 **MotrixLab v0.2.0**（`c84d382b8c907e0ec288db4437339b43e9201494`）。最新主分支的接口已经改变，请勿把本作业代码直接安装到新版。Python 3.10，Windows/Linux 均可选择 Torch CPU；本次使用 Windows。

以下为 Windows PowerShell 示例。在官方源码根目录创建本项目专用环境，无需修改全局 Python：

```powershell
git clone --branch v0.2.0 https://github.com/Motphys/MotrixLab.git MotrixLab-course
cd MotrixLab-course
git lfs pull --include="motrix_envs/src/motrix_envs/common/motphys-ground.png"
uv venv --python 3.10 .venv
uv pip install --python .venv/Scripts/python.exe "torch==2.7.0" --index https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe "motrixsim==0.6.0b1" "skrl==1.4.3" "gymnasium==1.1.1" "python-abc==0.3.0" "tensorboard==2.21.0" "absl-py==2.1.0" "matplotlib==3.10.9" pillow tzdata
uv pip install --python .venv/Scripts/python.exe --no-deps -e ./motrix_envs -e ./motrix_rl
```

Linux 将上面的 `.venv/Scripts/python.exe` 改为 `.venv/bin/python`。选择 CPU 版 Torch 是为了适应无 NVIDIA GPU 的电脑；没有安装 CUDA、JAX 和训练无关的音视频包。

官方仓库中部分资源由 Git LFS 管理。CartPole 渲染需要真实的 `motphys-ground.png`，不能把 LFS 指针文本当成 PNG。该文件预期 SHA-256 为 `c1ed597194fe5c3cf8b46286568a9520be2af547b1f6be219c439956fdbdbfaf`。

下面假设作业仓库位于 `../MotrixLab`，请按实际路径修改。

### simple_walker 接入和训练

```powershell
uv run --no-sync ../MotrixLab/week2/code/install_simple_walker.py .
uv run --no-sync scripts/train.py --env simple_walker --sim-backend np --train-backend torch --num-envs 4 --seed 42
```

### CartPole 训练、曲线和播放

```powershell
uv run --no-sync ../MotrixLab/week2/code/train_cartpole.py
uv run --no-sync tensorboard --logdir ../MotrixLab/week2/runs/cartpole --host 127.0.0.1 --port 6006
uv run --no-sync scripts/play.py --env cartpole --sim-backend np --num-envs 1 --seed 42 --policy ../MotrixLab/week2/runs/cartpole/26-10-25_18-59-35-304424_PPO/checkpoints/best_agent.pt
uv run --no-sync ../MotrixLab/week2/code/evaluate_cartpole.py ../MotrixLab/week2/runs/cartpole/26-10-25_18-59-35-304424_PPO/checkpoints/best_agent.pt
```

重新训练会在当前工作目录产生新的 `runs/cartpole/<时间戳>_PPO`，不会覆盖提交的原始实验；要分析新实验，修改 TensorBoard 的 `--logdir` 和播放的 `--policy` 路径。

本次训练实际使用的独立环境位于 `D:/Xbotics/motrix-venv`，命令和完整安装版本记录在 `evidence/commands.md`、`evidence/environment.txt`。选择题第 7 讲前两题与该版本实际参数默认值的差异已在答案中明确说明。
