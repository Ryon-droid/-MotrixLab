# 本次实际执行与结果

日期：2026-11-08（Asia/Shanghai）。宿主：Windows，Python 3.10.20。

## 源码

从官方 `v0.3.0` 标签读取，提交为 `e1421d10559c825e72bab0f3cbb517c333d67e16`。使用 `git show <commit>:<path>` 提取原始 blob；`source_manifest.json` 保存每个文件的 SHA-256。完整官方仓库的无关 LFS 资源没有下载，CartPole 地面纹理使用已验证的真实 PNG，SHA-256 为 `c1ed597194fe5c3cf8b46286568a9520be2af547b1f6be219c439956fdbdbfaf`。

```powershell
python code/prepare_sources.py D:/Xbotics/MotrixLab-v0.3.0
```

## 独立环境

在 `week3` 下，用 `uv.lock` 创建新的 `.venv` 并实际执行，而非复用第二周的旧 MotrixSim：

```powershell
uv run python code/run_cartpole.py --headless --steps 2000 --report evidence/balance_run.json
uv run python code/run_cartpole.py --headless --mode random --steps 1000 --report evidence/random_run.json
uv run python code/verify_cartpole.py
```

- 默认平衡控制：2000 步、两个完整回合，每回合 1000 步，均为时间截断，失败回合数 0。LQR 控制器来自实际模型的数值线性化，没有训练或加载 RL 权重。
- 随机控制：1000 步内完成 17 个失败回合，最后一个未结束回合已走 100 步，说明失败和自动重置路径被实际触发。
- 规则检查：观测顺序、空间、时钟、严格边界、NaN 角度分支、1000 步超时、失败当步奖励、自动重置、1024 个初始化样本、正负动作限幅，全部通过。NaN 分支只对 `update_state()` 输入做定向检查，没有把 NaN 送入物理解算器。

另实际启动显示窗口并走 100 步：

```powershell
python code/run_cartpole.py --steps 100 --report D:/Xbotics/tmp/week3_gui_smoke.json
```

## 效果图与文档

```powershell
python code/capture_cartpole.py
python code/build_report.py
```

相机图使用 MotrixSim 0.7.0 headless renderer，添加固定世界坐标相机，设置关节状态后显式执行正向运动学；没有截图或操纵巧工软件。第一次视觉检查发现异步渲染尚未加载完以及关节图形尚未更新，修正加载等待和正向运动学后，重新生成并目视检查全部三张最终图。最终采图日志保存在 `capture.log`，模型文件和物理参数未修改。

PDF 使用 ReportLab 生成，以 PDFium 渲染逐页检查；源码图保留真实文件名、行号、提交号。文档生成依赖与仿真依赖分离。
