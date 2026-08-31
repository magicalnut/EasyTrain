#ifndef TRAININGWORKER_H
#define TRAININGWORKER_H
#pragma once

#include <QObject>

#undef slots

#include "model/model.h"
#include <atomic>

class TrainingWorker : public QObject
{
    Q_OBJECT
public:
    TrainingWorker(Model* model, torch::Tensor x, torch::Tensor y,torch::Device dev, int batch, int epochs);
    void run();    // 训练主循环（worker 线程里跑）
    void pause();  // 正常暂停：置 pause_，跑完当前 epoch 停
    void abort();  // 急停：置 abort_，跑完当前 batch 停

signals:
    void lossReady(float loss);
    void epochDone(int epoch);
    void finished();

private:
    Model* model_;            // 非拥有，生命周期由 MainWindow 保证
    torch::Tensor x_, y_;
    torch::Device dev_;
    int batch_, epochs_;
    std::atomic<bool> pause_{false};   // 正常暂停：epoch 边界检查
    std::atomic<bool> abort_{false};   // 急停：batch 边界检查（传给 fit）
};

#endif // TRAININGWORKER_H
