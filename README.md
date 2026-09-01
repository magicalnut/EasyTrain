# EasyTrain

无代码（No-Code）深度学习训练桌面应用：用鼠标搭网络、选数据、点开始就能训练，不需要写任何训练代码。

- 框架：Qt 6.5 Widgets + LibTorch 2.13 (cu126) + OpenCV
- 语言 / 标准：C++20
- 平台：Windows 11，MSVC 2022

## 当前功能

### 模型搭建（主从编辑）
- 三栏布局：左「层列表」增删层，中「详情」编辑每层参数，右「训练」配置数据与超参。
- 支持的层：`Conv2d`、`MaxPool2d`、`Linear`、`ReLU`、`Flatten`、`Dropout`。
- 输入形状 `[N, C, H, W]`：N 固定为动态 `-1`，C/H/W 可编辑。
- 编辑期 **JSON 是唯一真相源**；点「应用」才真正构建成 torch 模型（形状推断不合法会弹错）。
- 支持导入 / 导出模型结构 JSON。

### 数据
- 文件夹：每个子目录一个类别（按目录名排序编号）。
- CSV：每行 `path,label`（数字标签直接当类索引，字符串标签按出现顺序映射）。
- 一键「生成随机数据」（256 样本 / 10 类），用于冒烟跑通全流程。

### 训练
- 设备：`cpu` / `cuda:N`。
- 超参：batch / lr / epoch（lr 训练中也可调，下个 epoch 生效）。
- 优化器：Adam（结构已预留换 sgd / adamw 的位子）。
- 损失：CrossEntropyLoss。
- 控制：开始 / 暂停（跑完当前 epoch 停）/ 急停（跑完当前 batch 停）。
- 实时 loss 曲线。

### 权重
- 保存 / 导入 `.pt` 权重（导入时结构不一致会报 key 不匹配）。

### 资源监视器
- 独立窗口：CPU 占用、内存占用、GPU 利用率、显存占用（GPU 数据经 `nvidia-smi` 读取）。

## 构建

### 依赖

| 依赖 | 版本 | 说明 |
|---|---|---|
| Qt | 6.5+ | Core、Widgets |
| LibTorch | 2.13.0 + cu126 | 解压到项目根目录 `libtorch/` |
| CUDA | 12.6 | 含对应 cuDNN |
| OpenCV | 源码 | 放到项目根目录 `opencv/`，FetchContent 源码编译 |
| MSVC | 2022 17.8+ | 需 C++20 |

### 步骤

用 Qt Creator（MSVC 2022 kit）直接打开 `CMakeLists.txt` 构建即可；或命令行：

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

构建后会自动把 `opencv_world.dll`、LibTorch 的 DLL 复制到 exe 旁边，并执行 `windeployqt` 部署 Qt 运行时，exe 可脱离 Qt Creator 独立运行。

### 构建注意事项

- LibTorch 2.13 头文件用到 C++20 语法（`__VA_OPT__` 等），必须开 `/std:c++20` 与 MSVC `/Zc:preprocessor`（`CMakeLists.txt` 已配好）。
- OpenCV 只编译 `core/imgproc/imgcodecs/highgui` 四个模块，并关闭 Qt 后端（`WITH_QT OFF`），避免拉进一堆用不到的 Qt 模块。

## 使用流程

1. **搭网络**：点「加层」→ 在详情页设每层参数 → 设输入 C/H/W → 点「应用」。
2. **准备数据**：「加载数据集」（文件夹或 CSV）或「生成随机数据」。
3. **训练**：选设备、batch / lr / epoch → 点「开始」；需要时「暂停」或「急停」。
4. **保存 / 导入权重**：`.pt`。

## 项目结构

```
EasyTrain/
├── main.cpp                  # 入口
├── mainwindow.h/.cpp         # 主窗口：三栏布局 + 训练接线
├── monitor.h/.cpp            # 资源监视器（CPU/内存/GPU，3 线程 + nvidia-smi）
├── plot_widget.h/.cpp        # 折线图控件（loss 曲线、监视器曲线）
├── trainingworker.h/.cpp     # 训练 worker（moveToThread，暂停/急停）
├── model/
│   ├── model.h/.cpp          # Model：组合 torch 模块、fit()、优化器、权重
│   ├── model_json.h/.cpp     # JSON ↔ Model 互转
│   ├── layer_config.h        # 层配置（类型 + 参数）
│   ├── layer_factory.h/.cpp  # 层配置 → torch::nn 模块
│   ├── shape.h/.cpp          # 形状推断（承重墙）
│   └── dataset.h/.cpp        # 数据集加载（文件夹 / CSV / 随机）
├── doc/                      # 设计方案文档
├── libtorch/                 # LibTorch（需自行放置）
├── opencv/                   # OpenCV 源码（需自行放置）
└── CMakeLists.txt
```

## 已知限制 / 待办

- **数据增强**：尚未实现（规划 v2，OpenCV 在线增强：翻转 / 裁剪 / 旋转 / 亮度）。
- **数据加载**：当前全量加载进内存 + 整份搬上 GPU（v1 同步），大数据集会卡 UI，且显存占用与 batch 无关。规划 v2 改为路径 + 懒加载 + 可选预取线程。
- **优化器**：目前只有 Adam，结构已预留 sgd / adamw。
- **训练中限制**：训练期间除 lr 与暂停 / 急停外，其余控件禁用。
