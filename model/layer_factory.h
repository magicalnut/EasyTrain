#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H
#pragma once
#include "layer_config.h"

#include <torch/torch.h>

torch::nn::AnyModule create_layer(const LayerConfig& L);

#endif // LAYER_FACTORY_H
