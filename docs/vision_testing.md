# 雷达站视觉实拍检查

从项目目录运行下面的命令。先关闭 MVS 客户端或其它相机预览；相机按独占方式打开。

```bash
cd /home/meko/JGRadar
.venv/bin/python tools/vision_preview.py --log out/vision_live.jsonl
```

默认使用真实 EGA ONNX 权重及 MobileNet 数字模型，曝光 10000 微秒、增益 23 dB，退出时恢复相机参数。
按 Q 或 Esc 退出。可用 `--exposure-us` 和 `--gain-db` 调整室内亮度；明亮场景应降低增益或曝光，避免装甲灯带过曝。

绿色 CANDIDATE 框是一级模型的候选目标，不代表已确认的车辆。EGA 一级 ONNX
元数据只定义一个 `item` 类别；它在独立装甲板、人员和其它背景物体上仍可能误检。
装甲框的 RED / BLUE / DEAD 来自二级装甲颜色类别，只有二级实际输出时才显示。
已配置数字模型时同时显示装甲数字并写入 digits 日志；本窗口不运行持续跟踪和场地定位，
彩色装甲加数字仍不能代替最终机器人身份、跟踪和定位的验收。
顶部 FPS 是实际处理速度，infer 是一次检测级联的耗时。GPU/GPU 表示车辆与装甲
会话都启用了 CUDA；CPU 表示该会话实际使用 CPU。实时相机的 age 是 SDK
读到该帧后到推理完成的时间，不包括传感器曝光到 USB 交付之前的时间。
采集线程持续取流，只保留最新帧；处理较慢时会跳过中间帧，防止画面累积延迟。

默认 `cascade` 模式只在一级候选框内检测装甲。**独立装甲板测试**应使用下面的
`independent` 模式：车身与装甲分别检测，一级没有车框也会运行装甲模型。

```bash
.venv/bin/python tools/vision_preview.py --mode independent --log out/vision_independent.jsonl
```

只检查装甲可选 `--mode armor`。独立模式默认把整图缩到装甲模型的 192×192
输入，适合目前这种装甲较大的近距离画面；远处小装甲需要指定区域，不能期待
整张 1200 万像素图缩小后仍保留每一块装甲的细节。
例如 `--armor-roi 0 0 0.5 0.6` 仅检查左上区域；四个值是宽高比例。
独立装甲日志的 `car_index=-1`，没有推断所属车辆或机器人身份。

已有图片可复核，无需占用相机：

```bash
.venv/bin/python tools/vision_preview.py --image /path/to/image.png \
  --mode independent --no-display --save-frame out/independent_result.png
```

## 建议顺序

1. **取流与调焦**：先把完整车辆放在实际使用距离，对准车体和装甲板，调整镜头至轮廓与数字清晰。
   这一步可以用帧率较高的纯相机预览，然后按 Q 关闭，再启动检测预览。

   ```bash
   .venv/bin/python tools/camera_preview.py --exposure-us 10000 --gain-db 23
   ```

   连续稳定性可用下式检查；9000 帧在约 30 fps 时接近五分钟，实际时长按输出帧率计算。

   ```bash
   .venv/bin/python tools/camera_preview.py --exposure-us 10000 --gain-db 23 --no-display --frames 9000
   ```

2. **整车与装甲检测**：先测一台静止车辆，再测运动、转向、两车并行或交叉、短时遮挡。
   在近、中、远三档测试，包括实际最远使用距离；每档至少保留一段带框录像或逐帧日志。
   重点看框是否贴住真实车辆、装甲板，红蓝是否正确，以及转向或遮挡时是否长期丢框。
   用空背景、人员和普通物体做反例，记录错误的候选框，不能只看正例有没有框。
   日志的每个数组元素是一次检测，不是唯一车辆编号。

   当前级联先检测整车，再在车框内检测装甲，因此只把一张装甲图案放在镜头前，可能完全没有框。
   先用 `--mode independent` 或 `--mode armor` 验证独立装甲，再用默认模式验证安装了装甲板的完整车辆。

3. **身份与跟踪**：分别验证红蓝同号码、不同号码、转身、交叉及遮挡后重现。
   完整验收应能持续区分 R1/B1 等身份，并避免换车时继承错误身份。
   EGA 的 `.pth` 数字模型现通过 MobileNet 后端接入，Stage A 已将同车装甲颜色和数字组合成身份。
   需要可选依赖 `infer-digit`（torch/torchvision/Pillow）；配置中的数字尺寸为 64。
   数字模型第 4 类映射哨兵 7，第 5 类 Q 不生成机器人身份。检测预览只展示数字分类结果；
   正式身份、重捕获和定位应继续使用完整流水线验收。

4. **场地定位**：调焦固定后使用实测内参，在你自己的安装位置求外参，并重新生成像素到场地的映射表。
   同款相机不代表安装位置相同，EGA 录像的外参只适用于那段录像。
   用没有参与求外参的已知位置检查定位误差，覆盖近端、远端、边缘，以及高地或坡道。
   平面映射只验证平地；高地/坡道需验证网格映射。
   `tools/calibrate_extrinsics.py`、`tools/validate_alignment.py` 和 `tools/build_lookup_map.py` 提供现有入口。

5. **实时输出**：上述功能成立后，检查结果帧龄、丢弃比例、断流后的行为和输出身份/坐标。
   当前配置 `pipeline.max_commit_age_s=0.15` 会丢弃超过 150 ms 的帧；不要通过放宽门限掩盖推理过慢。
   20 fps 对应约 50 ms 的每帧处理预算。相机能输出约 30 fps，并不证明检测流水线也能达到该速度。

## 2026-10-10 已执行的检查

- EGA 录像的 12 帧检测与平面投影可运行，已生成 `out/vision_check_20261010/detection_report.html` 和拼图。
  这是执行链路检查，图中存在人员被误框为车辆的情况，不能据此宣称检测准确率通过。
- 新检测预览使用真实录像处理 6 帧，累计 43 个车辆检测框、15 个装甲检测框，颜色输出包括 red/blue/dead。
  平均检测级联耗时 603.2 ms。
- 新检测预览使用当前 USB3 相机处理 10 帧，采集序号从 0 推进至 172。
  当前实拍内容是桌面，未出现完整测试车辆，车辆与装甲输出均为零，不能据此验收实车识别。
  当时使用 CPU，平均检测级联耗时 641.8 ms，平均软件帧龄 659.0 ms，未达到 150 ms 的实时输出要求。
- 用户后续实拍中，独立装甲板和车体均被一级模型框出，而二级未确认装甲。
  旧预览把所有一级输出写成 CAR，现已改为 CANDIDATE；这纠正了显示，未修复模型误检。
- 已在项目虚拟环境将 CPU 推理库替换为 `onnxruntime-gpu==1.26.0`，复用本机
  CUDA 12.8 / cuDNN 9，并修复 cuDNN 位于用户 site-packages 时的加载路径。
  RTX 4060 Laptop 上的车辆、装甲会话都确认启用了 CUDAExecutionProvider。
- 同一录像前 24 帧：CPU 平均 587.7 ms，GPU 平均 94.7 ms，约快 6.2 倍。
  GPU 共输出 175 个一级候选框、65 个装甲框；结果见 `out/vision_check_20261010/gpu_video.jsonl`。
- GPU 实拍 60 帧：当前相机朝向没有测试车辆的地面，平均推理 68.7 ms，平均软件
  帧龄 85.2 ms、最大 111.4 ms。推理耗时对应约 14.6 fps，不等于相机的采集帧率。
  这段输入没有目标，不能代替多车场景的速度与准确率验收。
- 用户第二张截图的独立检查：一级无候选框，独立装甲模型能输出蓝色装甲框，
  置信度 0.757。排除查看器边框后检测约 0.791，装甲周围区域检测约 0.884。
  这证明该输入中的独立装甲被原来的车框门控挡住，不代表车身识别已经通过。
  结果见 `out/vision_check_20261010/independent_screenshot.png` 和对应 JSONL。
  这些是截图复核，原图重采样和画面裁切会影响置信度，不能直接作为原始实拍精度统计。

## GPU 环境

当前虚拟环境已配置，无需再次安装。重新创建环境时，CPU 与 GPU 版 ONNX Runtime
应二选一。CUDA 12 / cuDNN 9 使用以下 GPU extra；本机已验证的版本是 1.26.0。

```bash
.venv/bin/python -m pip uninstall -y onnxruntime
.venv/bin/python -m pip install '.[infer-onnx-gpu]'
```

启动时检查两行模型就绪输出均包含 CUDAExecutionProvider。若只有 CPUExecutionProvider，
说明没有启用 GPU，应检查运行库和设备访问，不能只凭已安装 GPU 包判断加速已生效。

录像检测复核命令：

```bash
.venv/bin/python tools/vision_preview.py \
  --video '/home/meko/桌面/EGA_Radar_Replay(2)/replay_sample/camera/camera_record.mp4' \
  --no-display --max-frames 12 --save-frame out/vision_video.png --log out/vision_video.jsonl
```

需要按场地检查录像投影时，用已有工具和录像自带的标定：

```bash
.venv/bin/python tools/render_overlay.py \
  --config config/default.yaml --config config/ega_replay.yaml \
  --every 1 --max-frames 24 --html out/vision_overlay.html --show
```
