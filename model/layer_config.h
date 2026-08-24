#ifndef LAYER_CONFIG_H
#define LAYER_CONFIG_H
#pragma once

#include <string>
#include <optional>

class LayerConfig {
public:
    std::string name;
    std::string type;

    // Conv2d / MaxPool2d
    std::optional<int64_t> in_channels;   // nullopt = 交给形状推断自动回填
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t padding = 0;
    int64_t dilation = 1;

    // Linear
    std::optional<int64_t> in_features;   // nullopt = 自动回填
    int64_t out_features = 0;

    // Dropout
    double p = 0.5;


    static LayerConfig conv(std::string name, int64_t out, int64_t k,int64_t s = 1, int64_t p = 0) {
        LayerConfig L;
        L.name = name; L.type = "conv2d";
        L.out_channels = out; L.kernel = k; L.stride = s; L.padding = p;
        return L;
    }
    static LayerConfig pool(std::string name, int64_t k, int64_t s = 0) {
        LayerConfig L;
        L.name = name; L.type = "maxpool2d";
        L.kernel = k; L.stride = (s == 0 ? k : s);
        return L;
    }
    static LayerConfig linear(std::string name, int64_t out) {
        LayerConfig L;
        L.name = name; L.type = "linear"; L.out_features = out;
        return L;
    }
    static LayerConfig relu(std::string name) {
        LayerConfig L; L.name = name; L.type = "relu"; return L;
    }
    static LayerConfig flatten(std::string name) {
        LayerConfig L; L.name = name; L.type = "flatten"; return L;
    }
    static LayerConfig dropout(std::string name, double prob = 0.5) {
        LayerConfig L; L.name = name; L.type = "dropout"; L.p = prob; return L;
    }
};

#endif // LAYER_CONFIG_H
