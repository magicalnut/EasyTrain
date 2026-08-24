#include "shape.h"
#include <stdexcept>

std::vector<Shape> infer_shapes(const Shape& input, std::vector<LayerConfig>& layers) {
    std::vector<Shape> ins;
    Shape cur = input;

    for (auto& L : layers) {
        ins.push_back(cur);

        if (L.type == "conv2d") {
            if (cur.size() != 4) throw std::runtime_error(L.name + ": Conv2d 需要 4D 输入 [N,C,H,W]");
            if (L.in_channels && L.in_channels != cur[1])
                throw std::runtime_error(L.name + ": 填写的 in_channels 与上游通道数不符");
            if (!L.in_channels) L.in_channels = cur[1];       // 自动回填
            int64_t Ho = (cur[2] + 2*L.padding - L.dilation*(L.kernel-1) - 1) / L.stride + 1;
            int64_t Wo = (cur[3] + 2*L.padding - L.dilation*(L.kernel-1) - 1) / L.stride + 1;
            cur = {cur[0], L.out_channels, Ho, Wo};
        }
        else if (L.type == "maxpool2d") {
            if (cur.size() != 4) throw std::runtime_error(L.name + ": MaxPool2d 需要 4D 输入");
            int64_t Ho = (cur[2] + 2*L.padding - L.dilation*(L.kernel-1) - 1) / L.stride + 1;
            int64_t Wo = (cur[3] + 2*L.padding - L.dilation*(L.kernel-1) - 1) / L.stride + 1;
            cur = {cur[0], cur[1], Ho, Wo};
        }
        else if (L.type == "relu" || L.type == "dropout") {
            // 形状不变
        }
        else if (L.type == "flatten") {
            int64_t d = 1;
            for (size_t i = 1; i < cur.size(); ++i) d *= cur[i];
            cur = {cur[0], d};
        }
        else if (L.type == "linear") {
            if (cur.size() != 2)
                throw std::runtime_error(L.name + ": Linear 需要 2D 输入 [N,D]，请先加 Flatten");
            int64_t d = cur[1];
            if (L.in_features && L.in_features != d)
                throw std::runtime_error(L.name + ": 填写的 in_features 与上游不符");
            if (!L.in_features) L.in_features = d;
            cur = {cur[0], L.out_features};
        }
        else {
            throw std::runtime_error("未知层类型: " + L.type);
        }
    }
    return ins;
}