#include "trainingworker.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

TrainingWorker::TrainingWorker(Model* model, const Dataset* dataset, int C, int H, int W,AugConfig aug, torch::Device dev, int batch, int epochs)
    : model_(model)
    , dataset_(dataset)
    , C_(C)
    , H_(H)
    , W_(W)
    , aug_(aug)
    , dev_(dev)
    , batch_(batch)
    , epochs_(epochs) {}

void TrainingWorker::run() {
    model_->prepareTraining();
    model_->to(dev_);

    torch::nn::CrossEntropyLoss loss_fn;
    std::mt19937 rng(std::random_device{}());

    for (int e = 0; e < epochs_; ++e) {
        if (pause_.load()){
            break;
        }

        std::vector<size_t> idx(dataset_->size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);

        double total = 0.0;
        int64_t seen = 0;
        for (size_t b = 0; b < idx.size(); b += batch_) {
            if (abort_.load()){
                break;
            }

            size_t cnt = std::min<size_t>(batch_, idx.size() - b);
            std::vector<torch::Tensor> xs;
            std::vector<int64_t> ys;
            xs.reserve(cnt);
            ys.reserve(cnt);
            for (size_t k = 0; k < cnt; ++k) {
                size_t i = idx[b + k];
                xs.push_back(sample_at(*dataset_, i, C_, H_, W_, aug_, true));
                ys.push_back(dataset_->labels_[i]);
            }
            auto xb = torch::stack(xs, 0).to(dev_);
            auto yb = torch::tensor(ys, torch::TensorOptions().dtype(torch::kLong)).to(dev_);

            float loss = model_->trainBatch(xb, yb, loss_fn);
            total += loss * (double)cnt;
            seen += (int64_t)cnt;
        }
        if (abort_.load()){
            break;
        }

        emit lossReady(static_cast<float>(total / std::max<int64_t>(seen, 1)));
        emit epochDone(e);
    }
    emit finished();
}

void TrainingWorker::pause() { pause_.store(true); }

void TrainingWorker::abort() { abort_.store(true); }
