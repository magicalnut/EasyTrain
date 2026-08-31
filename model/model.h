#ifndef MODEL_H
#define MODEL_H

#pragma once

#include "layer_config.h"
#include "shape.h"

#include <torch/torch.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

// 优化器配置（预留：v2 换类型只动 makeOptimizer 一个 case + 这里的字段）
struct OptimizerConfig {
    std::string type = "adam";   // 将来：sgd / adamw / rmsprop
    double lr = 1e-3;
    // 将来再加：momentum / weight_decay / betas…
};

struct Impl;   // 前置声明

class Model
{
public:
    Model();
    explicit Model(const Shape& input);
    ~Model();

    // —— 结构——
    void setInputShape(const Shape& input);          // 输入契约 [N,C,H,W]，N 可 -1
    void addLayer(const LayerConfig& layer);         // 追加一层，只记配置，不立刻建
    void clear();                                     // 清空所有层
    void build();                                     // 应用配置：推断形状→建模块→换 impl_→作废 optimizer
    // 读回配置（UI「刷新」用：Model → JSON）
    const Shape& inputShape() const;                  // 输入契约
    const std::vector<LayerConfig>& layers() const;   // 层配置（真相源）
    const OptimizerConfig& optimizerConfig() const;   // 优化器配置
    // TODO: void modifyLayer(size_t i, const LayerConfig& layer);

    // —— 前向 ——
    torch::Tensor forward(torch::Tensor x);           // 未 build 则先 build

    // —— 训练 ——
    float fit(const torch::Tensor& x, const torch::Tensor& y, int batch = 32,const std::atomic<bool>* stop = nullptr); // 一个 epoch，返回平均 loss
    void setOptimizer(const OptimizerConfig& cfg);    // 训练前设置/换优化器（只重置 optimizer_，不重建模块）
    void setLearningRate(double lr);                  // 训练中可改，下个 epoch 生效
    double learningRate() const;

    // —— 模式 / 设备 ——
    void to(torch::Device device);
    void train();
    void eval();

    // —— 持久化 ——
    void saveWeights(const std::string& path) const;  // state_dict → .pt
    void loadWeights(const std::string& path);

private:
    Shape input_;
    std::vector<LayerConfig> layers_;   // 真相源：保存用户原始意图，in_channels 保持 nullopt
    bool dirty_ = true;

    std::shared_ptr<Impl> impl_;
    torch::Device device_ = torch::kCPU;

    OptimizerConfig opt_cfg_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;   // 基类指针 → 换类型不换字段
    std::atomic<double> lr_{1e-3};

    void ensureBuilt();
    void ensureOptimizer();
    void applyLearningRate();
};

#endif // MODEL_H
