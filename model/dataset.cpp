#include "dataset.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace {

// 读图走二进制流 + imdecode：Windows 中文路径不踩 cv::imread 的窄字符坑
cv::Mat read_image(const fs::path& p) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("读图失败: " + p.string());
    }
    std::vector<uchar> buf((std::istreambuf_iterator<char>(ifs)),
                           std::istreambuf_iterator<char>());
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("解码失败: " + p.string());
    }
    return img;
}

bool is_image(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// 一张图 → resize 到 (W,H) → 转 C 通道 → [C,H,W] float 张量
torch::Tensor to_tensor(const cv::Mat& img, int C, int H, int W) {
    cv::Mat m;
    if (C == 1)       {
        cv::cvtColor(img, m, cv::COLOR_BGR2GRAY);
    }
    else if (C == 3)  {
        m = img;        // 已是 BGR 三通道
    }
    else{
        throw std::runtime_error("不支持的通道数 C（v1 只支持 1 或 3）");
    }

    cv::resize(m, m, cv::Size(W, H));
    m.convertTo(m, CV_32F, 1.0 / 255.0);

    torch::Tensor t;
    if (C == 1) {
        t = torch::from_blob(m.data, {H, W}, torch::kFloat32);
        t = t.unsqueeze(0).clone();                  // [1,H,W]；clone：from_blob 不拥有数据
    }
    else {
        t = torch::from_blob(m.data, {H, W, 3}, torch::kFloat32);
        t = t.permute({2, 0, 1}).clone();            // [3,H,W]
    }
    return t;
}

thread_local std::mt19937 g_rng(std::random_device{}());

// 随机增广 + resize 到 (W,H)。train=false 只 resize。
cv::Mat augment(const cv::Mat& img, const AugConfig& aug, int H, int W, bool train) {
    cv::Mat m = img;
    if (train) {
        if (aug.random_crop) {
            int pad = std::max(1, H / 8);
            cv::copyMakeBorder(m, m, pad, pad, pad, pad, cv::BORDER_REFLECT);
            std::uniform_int_distribution<int> d(0, 2 * pad);
            int x = d(g_rng), y = d(g_rng);
            m = m(cv::Rect(x, y, img.cols, img.rows)).clone();
        }
        if (aug.hflip) {
            std::uniform_int_distribution<int> d(0, 1);
            if (d(g_rng)) cv::flip(m, m, 1);
        }
        if (aug.rotate_deg > 0) {
            std::uniform_real_distribution<double> d(-aug.rotate_deg, aug.rotate_deg);
            double ang = d(g_rng);
            cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(m.cols / 2.f, m.rows / 2.f), ang, 1.0);
            cv::warpAffine(m, m, R, m.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
        }
        if (aug.brightness > 0 && m.channels() == 3) {
            std::uniform_real_distribution<double> d(1.0 - aug.brightness, 1.0 + aug.brightness);
            m.convertTo(m, -1, d(g_rng), 0);
        }
    }
    cv::resize(m, m, cv::Size(W, H));
    return m;
}

}

Dataset load_dataset(const std::wstring& src, bool is_csv) {
    fs::path root(src);
    std::vector<std::wstring> paths;
    std::vector<int64_t> labels;
    int64_t nclasses = 0;

    if (!is_csv) {
        std::vector<fs::path> class_dirs;
        for (const auto& e : fs::directory_iterator(root)){
            if (e.is_directory()) {
                class_dirs.push_back(e.path());
            }
        }
        std::sort(class_dirs.begin(), class_dirs.end());
        nclasses = (int64_t)class_dirs.size();
        if (nclasses == 0) {
            throw std::runtime_error("没有找到任何类别子目录");
        }

        for (int64_t c = 0; c < nclasses; ++c) {
            std::vector<fs::path> files;
            for (const auto& e : fs::directory_iterator(class_dirs[c])){
                if (e.is_regular_file() && is_image(e.path())) {
                    files.push_back(e.path());
                }
            }
            std::sort(files.begin(), files.end());
            for (const auto& f : files) {
                paths.push_back(f.wstring());
                labels.push_back(c);
            }
        }
    }
    else {
        std::ifstream in(root);
        if (!in) {
            throw std::runtime_error("打开 CSV 失败: " + root.string());
        }
        fs::path csv_dir = root.parent_path();
        std::map<std::string, int64_t> str2idx;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            size_t comma = line.find(',');
            if (comma == std::string::npos) {
                continue;
            }
            std::string path  = trim(line.substr(0, comma));
            std::string label = trim(line.substr(comma + 1));
            if (path.empty() || label.empty()) {
                continue;
            }

            fs::path img_path(path);
            if (img_path.is_relative()) {
                img_path = csv_dir / img_path;
            }

            int64_t idx;
            bool numeric = std::all_of(label.begin(), label.end(), [](unsigned char c){
                return std::isdigit(c) || c == '-';
            });
            if (numeric) {
                idx = std::stoll(label);
            }
            else {
                auto it = str2idx.find(label);
                if (it == str2idx.end()) {
                    idx = (int64_t)str2idx.size();
                    str2idx[label] = idx;
                }
                else {
                    idx = it->second;
                }
            }
            paths.push_back(img_path.wstring());
            labels.push_back(idx);
        }
        nclasses = 0;
        for (auto y : labels) {
            nclasses = std::max(nclasses, y + 1);
        }
    }

    if (paths.empty()) {
        throw std::runtime_error("数据集为空");
    }
    return Dataset{std::move(paths), std::move(labels), nclasses};
}

Dataset random_dataset(const std::wstring& out_dir, int N, int C, int H, int W, int classes) {
    fs::path root(out_dir);
    fs::create_directories(root);
    for (int c = 0; c < classes; ++c){
        fs::create_directories(root / ("class" + std::to_string(c)));
    }

    for (int i = 0; i < N; ++i) {
        int c = i % classes;
        cv::Mat m(H, W, (C == 1 ? CV_8UC1 : CV_8UC3));
        cv::randu(m, 0, 255);
        fs::path p = root / ("class" + std::to_string(c)) / (std::to_string(i) + ".bmp");
        cv::imwrite(p.string(), m);
    }
    return load_dataset(out_dir, false);// 复用文件夹加载，返回懒 Dataset
}

torch::Tensor sample_at(const Dataset& d, size_t i, int C, int H, int W,const AugConfig& aug, bool train) {
    cv::Mat img = read_image(fs::path(d.paths_[i]));// BGR

    cv::Mat m;
    if (C == 1){
        cv::cvtColor(img, m, cv::COLOR_BGR2GRAY);
    }
    else if (C == 3){
        m = img;
    }
    else{
        throw std::runtime_error("不支持的通道数 C（只支持 1 或 3）");
    }

    m = augment(m, aug, H, W, train);
    m.convertTo(m, CV_32F, 1.0 / 255.0);

    if (C == 1) {
        auto t = torch::from_blob(m.data, {H, W}, torch::kFloat32);
        return t.unsqueeze(0).clone();// [1,H,W]；clone：from_blob 不拥有数据
    }
    auto t = torch::from_blob(m.data, {H, W, 3}, torch::kFloat32);
    return t.permute({2, 0, 1}).clone();// [3,H,W]
}