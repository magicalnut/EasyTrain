#include "layer_factory.h"

torch::nn::AnyModule create_layer(const LayerConfig& L) {
    if (L.type == "conv2d"){
        return torch::nn::AnyModule(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(L.in_channels.value(), L.out_channels, L.kernel).stride(L.stride).padding(L.padding).dilation(L.dilation)));
    }
    if (L.type == "maxpool2d"){
        return torch::nn::AnyModule(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(L.kernel).stride(L.stride).padding(L.padding)));
    }
    if (L.type == "relu"){
        return torch::nn::AnyModule(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
    }
    if (L.type == "flatten"){
        return torch::nn::AnyModule(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)));
    }
    if (L.type == "dropout"){
        return torch::nn::AnyModule(torch::nn::Dropout(torch::nn::DropoutOptions(L.p)));
    }
    if (L.type == "linear"){
        return torch::nn::AnyModule(torch::nn::Linear(torch::nn::LinearOptions(L.in_features.value(), L.out_features)));
    }
    throw std::runtime_error("未知层类型: " + L.type);
}