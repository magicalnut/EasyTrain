#include "test.h"

#include <torch/torch.h>
#include <QDebug>

Test::Test() {
    for (size_t i = 0; i < layers.size(); ++i) {
        qDebug()<<layers[i].name << "  in: [";
        for (auto d : ins[i]) qDebug() << d << " ";
        qDebug() << "]\n";
    }

    // 3) 搭模型 + forward 验证
    DAGModel model(layers);
    auto x = torch::randn({1, 3, 32, 32});
    std::cout << "output shape: " << model.forward(x).sizes() << "\n";

    // 4) 训练（随机数据）
    std::vector<float> losses = train_for_demo(model,200,32);
    for (size_t i = 0; i < losses.size(); ++i)
        qDebug() << "step " << i * 10 << " loss=" << losses[i] << "\n";
}

Test::~Test(){
}
