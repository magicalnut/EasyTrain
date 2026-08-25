#ifndef TEST_H
#define TEST_H

#include "layer_config.h"
#include "shape.h"
#include "dag_model.h"

#include <vector>

// 随机数据最小训练：验证 loss 下降 + 设备切换 + train/eval 机制
// 定义在头文件里、被多个 .cpp include，必须 inline，否则 LNK2005（每个 TU 各一份定义）。
inline std::vector<float> train_for_demo(DAGModel& model, int steps = 200, int batch = 32) {
    torch::Device dev = torch::cuda::is_available()
    ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    model.to(dev);
    model.train();

    torch::optim::Adam opt(model.parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::CrossEntropyLoss loss_fn;

    std::vector<float> losses;
    auto x = torch::randn({batch, 3, 32, 32}, torch::TensorOptions().device(dev));
    auto y = torch::randint(0, 10, {batch},
                            torch::TensorOptions().dtype(torch::kLong).device(dev));  // 标签必须 kLong
    for (int s = 0; s < steps; ++s) {

        opt.zero_grad();
        auto loss = loss_fn(model.forward(x), y);
        loss.backward();
        opt.step();
        if (s % 10 == 0) losses.push_back(loss.item<float>());
    }
    return losses;
}

class Test
{
public:
    Test();
    ~Test();
    // 1) 配置：in_channels / in_features 留空，交给形状推断
    std::vector<LayerConfig> layers = {
        LayerConfig::conv("conv1", 16, 3, /*stride*/1, /*padding*/1),
        LayerConfig::relu("relu1"),
        LayerConfig::pool("pool1", 2),   // stride 默认 = kernel = 2
        LayerConfig::flatten("flat"),
        LayerConfig::linear("fc", 10),
    };

    // 2) 形状推断：回填 + 校验
    Shape input = {1, 3, 32, 32};    // 输入契约：来自数据集配置（这里演示硬编码）
    std::vector<Shape> ins = infer_shapes(input, layers);


};

#endif // TEST_H
