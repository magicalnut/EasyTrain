#ifndef SHAPE_H
#define SHAPE_H
#pragma once

#include "layer_config.h"

#include <vector>
#include <cstdint>

using Shape = std::vector<int64_t>;

std::vector<Shape> infer_shapes(const Shape& input, std::vector<LayerConfig>& layers);

#endif // SHAPE_H
