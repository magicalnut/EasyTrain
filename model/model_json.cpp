#include "model_json.h"

Shape shape_from_json(const QJsonArray& a) {
    Shape s;
    for (const auto& v : a) s.push_back(v.toInteger());
    return s;
}

QJsonArray shape_to_json(const Shape& s) {
    QJsonArray a;
    for (auto v : s) a.append((qint64)v);
    return a;
}

LayerConfig layer_from_json(const QJsonObject& o) {
    LayerConfig L;   // 默认值已在 LayerConfig（stride=1/dilation=1/p=0.5…）
    L.name = o["name"].toString().toStdString();
    L.type = o["type"].toString().toStdString();
    // 只在 JSON 里有这个 key 才覆盖，否则保留默认值（.toInt() 缺 key 会返回 0，会冲掉默认值）
    if (o.contains("in_channels"))  L.in_channels  = o["in_channels"].toInteger();
    if (o.contains("out_channels")) L.out_channels = o["out_channels"].toInteger();
    if (o.contains("kernel"))       L.kernel       = o["kernel"].toInteger();
    if (o.contains("stride"))       L.stride       = o["stride"].toInteger();
    if (o.contains("padding"))      L.padding      = o["padding"].toInteger();
    if (o.contains("dilation"))     L.dilation     = o["dilation"].toInteger();
    if (o.contains("in_features"))  L.in_features  = o["in_features"].toInteger();
    if (o.contains("out_features")) L.out_features = o["out_features"].toInteger();
    if (o.contains("p"))            L.p            = o["p"].toDouble();
    return L;
}

QJsonObject layer_to_json(const LayerConfig& L) {
    QJsonObject o;
    o["name"] = QString::fromStdString(L.name);
    o["type"] = QString::fromStdString(L.type);
    if (L.in_channels) o["in_channels"] = (qint64)*L.in_channels;
    if (L.type == "conv2d") {
        o["out_channels"] = (qint64)L.out_channels;
        o["kernel"]       = (qint64)L.kernel;
        o["stride"]       = (qint64)L.stride;
        o["padding"]      = (qint64)L.padding;
        o["dilation"]     = (qint64)L.dilation;
    } else if (L.type == "maxpool2d") {
        o["kernel"]  = (qint64)L.kernel;
        o["stride"]  = (qint64)L.stride;
        o["padding"] = (qint64)L.padding;
    } else if (L.type == "linear") {
        if (L.in_features) o["in_features"] = (qint64)*L.in_features;
        o["out_features"] = (qint64)L.out_features;
    } else if (L.type == "dropout") {
        o["p"] = L.p;
    }
    return o;
}

OptimizerConfig optimizer_from_json(const QJsonObject& o) {
    OptimizerConfig c;
    c.type = o["type"].toString().toStdString();
    c.lr   = o["lr"].toDouble();
    return c;
}

QJsonObject optimizer_to_json(const OptimizerConfig& c) {
    QJsonObject o;
    o["type"] = QString::fromStdString(c.type);
    o["lr"]   = c.lr;
    return o;
}

void apply_json_to_model(const QJsonObject& root, Model& m) {
    m.clear();
    m.setInputShape(shape_from_json(root["input_shape"].toArray()));
    m.setOptimizer(optimizer_from_json(root["optimizer"].toObject()));
    for (const auto& v : root["layers"].toArray())
        m.addLayer(layer_from_json(v.toObject()));
    m.build();   // ← 承重墙：形状/类型错误在这里 throw
}

QJsonObject model_to_json(const Model& m) {
    QJsonObject root;
    root["input_shape"] = shape_to_json(m.inputShape());
    root["optimizer"]   = optimizer_to_json(m.optimizerConfig());
    QJsonArray layers;
    for (const auto& L : m.layers()) layers.append(layer_to_json(L));
    root["layers"] = layers;
    return root;
}
