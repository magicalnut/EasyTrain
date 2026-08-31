#ifndef MODEL_JSON_H
#define MODEL_JSON_H
#pragma once

#include <QJsonObject>
#include <QJsonArray>
#include <QString>

#undef slots

#include "model.h"

Shape shape_from_json(const QJsonArray& a);
QJsonArray shape_to_json(const Shape& s);

LayerConfig layer_from_json(const QJsonObject& o);   // 缺字段保留 LayerConfig 默认值
QJsonObject layer_to_json(const LayerConfig& L);     // 只写该类型相关的字段

OptimizerConfig optimizer_from_json(const QJsonObject& o);
QJsonObject optimizer_to_json(const OptimizerConfig& c);

void apply_json_to_model(const QJsonObject& root, Model& m);  // 应用：clear→setInputShape→setOptimizer→addLayer×N→build
QJsonObject model_to_json(const Model& m);                    // 刷新：读回 Model 当前配置

#endif // MODEL_JSON_H
