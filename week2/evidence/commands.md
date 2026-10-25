# 实际运行命令

日期：2026-10-25，Asia/Shanghai。以下是本机实际路径和参数，未省略 checkpoint 文件名。

## 运行版本

- 官方 MotrixLab v0.2.0：`c84d382b8c907e0ec288db4437339b43e9201494`。
- 官方源码目录：`D:/Xbotics/MotrixLab-course`。
- 独立 Python 环境：`D:/Xbotics/motrix-venv`。
- 作业提交目录：`D:/Xbotics/MotrixLab/week2`。
- CartPole 和 simple_walker 均通过官方环境注册表、官方 SKRL Torch PPO Trainer 实际训练。

## CartPole

工作目录为官方源码根目录：

```powershell
& 'D:\Xbotics\motrix-venv\Scripts\python.exe' -u 'D:\Xbotics\MotrixLab\week2\code\train_cartpole.py'
```

训练使用 `Trainer("cartpole", sim_backend="np", cfg_override=...)`。完整已解析参数在 `train.log`，共 4096 个批量步、32 个并行环境、131072 条环境转移。

训练结束后将 `D:/Xbotics/MotrixLab-course/runs/cartpole` 原样复制到本作业 `runs/cartpole`，包括原始 events 文件和全部模型。

## TensorBoard

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\Xbotics\motrix-venv'
uv run --no-sync tensorboard --logdir 'D:\Xbotics\MotrixLab\week2\runs\cartpole' --host 127.0.0.1 --port 6006
```

服务启动后读取 `/data/plugin/scalars/tags`，确认 reward、episode、policy loss、value loss 标签实际存在。原始数据导出命令：

```powershell
& 'D:\Xbotics\motrix-venv\Scripts\python.exe' 'D:\Xbotics\MotrixLab\week2\code\inspect_results.py'
```

`training_curves.png` 从原始 events 绘制，不是对 TensorBoard UI 的仿制。

`tensorboard_dashboard.png` 由本机 Edge 的 headless 截图功能访问上述实际本地 TensorBoard 服务生成，使用独立的临时浏览器目录，不读取日常浏览器的登录状态。截图参数为 1600×1600，页面路由为 `#scalars&tagFilter=Loss%7CReward%20%2F%20Total%20reward%20%5C(mean%5C)`；图表采用页面默认 0.6 的平滑系数。

## 原始 play.py 可视化播放

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\Xbotics\motrix-venv'
$env:OMP_NUM_THREADS='2'
uv run --no-sync scripts/play.py --env cartpole --sim-backend np --num-envs 1 --seed 42 --policy 'D:\Xbotics\MotrixLab\week2\runs\cartpole\26-10-25_18-59-35-304424_PPO\checkpoints\best_agent.pt'
```

实际显示 MotrixRender 窗口，截图为 `play_cartpole.png`。取得证据后主动停止该无限播放进程，训练本身正常完成。`play.log` 中旧 SKRL 导入方式的弃用提示和渲染器重复 loader 的警告不影响本次成功播放。

## 独立 checkpoint 评估

```powershell
& 'D:\Xbotics\motrix-venv\Scripts\python.exe' -u 'D:\Xbotics\MotrixLab\week2\code\evaluate_cartpole.py' 'D:\Xbotics\MotrixLab\week2\runs\cartpole\26-10-25_18-59-35-304424_PPO\checkpoints\best_agent.pt'
```

结果：2000 步，2 个完成回合，每回合长度 1000、回报 1000，所有观测有限。

## simple_walker

```powershell
& 'D:\Xbotics\motrix-venv\Scripts\python.exe' 'D:\Xbotics\MotrixLab\week2\code\install_simple_walker.py' 'D:\Xbotics\MotrixLab-course'
$env:UV_PROJECT_ENVIRONMENT='D:\Xbotics\motrix-venv'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
uv run --no-sync scripts/train.py --env simple_walker --sim-backend np --train-backend torch --num-envs 4 --seed 42
```

结果：8192 个批量步、4 个并行环境、32768 条环境转移，正常结束。另以 4 个环境执行 420 次零动作步进，通过注册、观测/动作维度、回合截断和有限值检查，见 `simple_walker_smoke.log`。
