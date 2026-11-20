# 推理依赖与测速评估（只出数字，未安装任何东西）

阶段 3 前的准备项：视觉推理现在 **954 ms/帧（约 1 fps）**，离实时差两个数量级。
这份文档只回答三件事：现在瓶颈在哪、装什么要多大代价、装完怎么用一条命令测出真数。

## 1. 环境事实（2026-11-21 实测）

| 项 | 实测值 |
| --- | --- |
| GPU | RTX 4060 Laptop（Max-Q），8 GB 显存，驱动 595.99.02，驱动侧支持 CUDA 13.2 |
| CUDA Toolkit | `/usr/local/cuda` 12.6.3（`libcublas.so.12` 等已在 `ldconfig` 里） |
| CPU | Ryzen 7 8845H（8 核 16 线程），30 GB RAM，`/home` 剩 74 GB |
| onnxruntime-gpu | **已装 1.26.0**，`get_available_providers()` = `Tensorrt / CUDA / CPU` |
| 缺的库 | `libcudnn.so.9`（ORT 报 `Require cuDNN 9.* and CUDA 12.*`，静默退到 CPU） |
| torch | **未安装**（数字分类模型 `mobilenet_v2` 需要） |
| torchvision | **已装 0.25.0+cu128** 但没有 torch —— 半装状态；deps 写死 `torch==2.10.0` |

## 2. CPU 实测（`tools/bench_inference.py`）

车级模型输入固定 1280×1280（导出时锁死，改不了分辨率）：

| 线程数 | ms/帧 |
| --- | --- |
| 2 | 955.8 |
| **4** | **675.0**（最好；不同轮次 675~1005，受功耗/温度影响） |
| 8 | 953.8 |
| 16 | 1164.5 |

装甲级模型输入动态，192px：

| 配置 | ms |
| --- | --- |
| 192px，4 线程 | 22.5 |
| 192px，8 线程 | 61.2 |
| 320px，8 线程 | 138.8 |
| batch=8（192px） | 41.3 ms/张 |
| batch=16 | 46.5 ms/张 |
| batch=32 | 35.3 ms/张 |

一辆车按 1 次车级 + 6 个装甲 ROI 估算：`675 + 6×22.5 ≈ 810 ms`，**约 1.2 fps**。
相机 30 fps 需要 33 ms/帧，10 fps 需要 100 ms/帧——CPU 路线差 8~25 倍，**不是调参能补上的**。

## 3. 两个可选动作（都还没做）

### A. 装 cuDNN 9（只为让 ORT 用上 GPU）

```zsh
sudo apt install libcudnn9-cuda-12      # 候选 9.28.1.4-1
```

- 下载 **518 MB**，安装后占用 **1.10 GB**（apt 元数据）
- 需要 sudo；不改 Python 环境
- 收益：`CUDAExecutionProvider` 能建起来 → 车级/装甲级的真实 GPU ms 立刻可测
- 风险：低。装的是系统库，不影响现有 CPU 路径

### B. 装 torch（只为数字分类模型）

`torchvision 0.25.0+cu128` 锁死 `torch==2.10.0`，而 PyPI 上 torch 2.10.0 cp312 单个轮子
**873 MB**，它还会拉以下 cu12 依赖（PyPI 轮子体积）：

| 包 | 大小 |
| --- | --- |
| nvidia-cudnn-cu12 9.10.2.21 | 674 MB |
| nvidia-cublas-cu12 12.8.4.1 | 567 MB |
| nvidia-nccl-cu12 2.27.5 | 307 MB |
| nvidia-cusparse / cusolver / cusparselt | 275 + 255 + 274 MB |
| nvidia-cufft / curand / nvshmem / nvrtc / triton / 其余 | 184 + 61 + 133 + 84 + 180 + 其余 |
| **依赖合计** | **约 3.04 GB** |

**总下载约 3.9 GB**，落盘按轮子解压后的经验值约 6~8 GB（这是**估算**，不是实测）。
另外这些是 cu12 轮子，与驱动 595 兼容，但会与系统里已有的 CUDA 12.6 头文件并存。

### C. 把数字模型导出成 ONNX（不用在本机装 torch）

`.pth` 只要在一台有 torch 的机器上导一次成 `.onnx`，运行时本机就只需要
`onnxruntime`——符合"运行时零重依赖"的项目取向。代价：需要在别处借一次 torch 环境。

## 4. 建议（等你拍板）

1. **先 A**：518 MB 换一个能测的 GPU 数字，性价比最高。装完一条命令见第 5 节。
2. 数字模型走 **C**（一次性导出）而不是 B；B 的 3.9 GB 只为跑一个 64px 的分类器，不划算。
3. 在拿到 GPU 实测之前，**任何"GPU 上大概多少 ms"的说法都不要进文档/论文**——
   这台机器现在没有实测值，估算值不要冒充。

## 5. 装完怎么测（一条命令）

```zsh
# 全部 provider 自动挑（装好 cuDNN 后会包含 CUDA）
.venv/bin/python tools/bench_inference.py --config config/vision_check.yaml --providers auto

# 指定对比：CPU 4 线程 vs CUDA
.venv/bin/python tools/bench_inference.py --model /path/car/best.onnx --providers cpu --threads 4
.venv/bin/python tools/bench_inference.py --model /path/car/best.onnx --providers cuda
```

输出是一条 JSON：`providers` 是**实际**用上的，`ms_per_frame` 是实测。若请求的
加速 provider 没建起来，工具会打 `fell_back_to_cpu: true` 并返回退出码 1——
因为 ORT 在这种情况下**不抛异常**，静默退到 CPU，读的人很容易把 CPU 的 ms 当成 GPU 的
（第一次跑 CUDA 时我就差点被这个骗过去）。

## 6. 当前结论

- 视觉推理**当前不可实时**（约 1 fps），这是阶段 3 验收前必须解决的前置项。
- 缺的库有明确候选（A 方案 518 MB），GPU 实测数字在装上之前**给不出**。
- 数字模型缺 torch，建议走导出 ONNX 的路线，不在本机装 3.9 GB。
