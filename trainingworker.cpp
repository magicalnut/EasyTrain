#include "trainingworker.h"

TrainingWorker::TrainingWorker(Model* model, torch::Tensor x, torch::Tensor y,torch::Device dev, int batch, int epochs)
    : model_(model)
    , x_(std::move(x))
    , y_(std::move(y))
    , dev_(dev),
    batch_(batch)
    , epochs_(epochs) {}

void TrainingWorker::run() {
    x_ = x_.to(dev_);          // 数据搬到设备：Model 不替调用方搬数据
    y_ = y_.to(dev_);
    model_->to(dev_);          // 模型已在 UI 迁过，这里兜一次保证一致

    for (int e = 0; e < epochs_; ++e) {
        if (pause_.load()) break;                           // 正常暂停：跑完上一个完整 epoch 就停
        float loss = model_->fit(x_, y_, batch_, &abort_);  // fit 批次间查 abort_
        if (abort_.load()) break;                           // 急停：不 emit 半截 loss
        emit lossReady(loss);
        emit epochDone(e);
    }
    emit finished();
}

void TrainingWorker::pause() { pause_.store(true); }
void TrainingWorker::abort() { abort_.store(true); }
