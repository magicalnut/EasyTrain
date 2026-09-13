#ifndef TRAININGWORKER_H
#define TRAININGWORKER_H
#pragma once

#include <QObject>

#undef slots

#include "model/model.h"
#include "model/dataset.h"
#include <atomic>

class TrainingWorker : public QObject
{
    Q_OBJECT
public:
    TrainingWorker(Model* model, const Dataset* dataset, int C, int H, int W,AugConfig aug, torch::Device dev, int batch, int epochs);
    void run();
    void pause();
    void abort();

signals:
    void lossReady(float loss);
    void epochDone(int epoch);
    void finished();

private:
    Model* model_;
    const Dataset* dataset_;
    int C_, H_, W_;
    AugConfig aug_;
    torch::Device dev_;
    int batch_, epochs_;
    std::atomic<bool> pause_{false};
    std::atomic<bool> abort_{false};
};

#endif // TRAININGWORKER_H
