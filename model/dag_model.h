#ifndef DAG_MODEL_H
#define DAG_MODEL_H
#pragma once

#include "layer_factory.h"

// 线性（顺序）happy-path 的 DAGModel。
// 结构上保留「topo_ 拓扑序 + name→module 映射」，将来加 Merge/分支只需扩展 forward 取数逻辑。
class DAGModel : public torch::nn::Module {
public:
    // layers 必须先经 infer_shapes 回填、形状合法
    explicit DAGModel(const std::vector<LayerConfig>& layers) {
        for (const auto& L : layers) {
            auto m = create_layer(L);
            register_module(L.name, m.ptr());       // 注册，to()/parameters()/train() 才能递归到
            modules_.emplace(L.name, std::move(m)); // 存起来给 forward 调用
            topo_.push_back(L.name);
        }
    }

    torch::Tensor forward(torch::Tensor x) {        // 不 override（Module 无虚函数 forward）
        for (const auto& name : topo_)
            x = modules_.at(name).forward(x);               // AnyModule::operator()
        return x;
    }

private:
    std::vector<std::string> topo_;
    std::unordered_map<std::string, torch::nn::AnyModule> modules_;
};

#endif // DAG_MODEL_H
