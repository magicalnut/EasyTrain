#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#pragma once

#include "monitor.h"

#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QCloseEvent>
#include <QSplitter>
#include <QListWidget>
#include <QStackedWidget>
#include <QLineEdit>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>
#include <QThread>

// Qt 的 qobjectdefs.h 把 `slots` 定义成空宏（#define slots），会和 libtorch 的
// Object::slots() 撞名：ivalue_inl.h:1585 的 `...& slots() const {` 被预处理成
// `...& () const {` → C2059/C2334。所有 Qt 头 include 完先 undef，再引入 torch。
// 注意不能 undef `signals`：本文件下面还在用 signals:。
#undef slots

#include "model/model.h"
#include "model/model_json.h"
#include "model/dataset.h"
#include "trainingworker.h"

class MainWindow : public QWidget
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
private:
    uint64_t layer_id;

    Monitor *monitor;

    QPushButton *button1;

    QVBoxLayout *mainlayout;
    QSplitter *hsplitter1;

    QWidget *widget1;
    QVBoxLayout *vlayout1_1;
    QHBoxLayout *hlayout1_1;   // 加层 / 删层
    QHBoxLayout *hlayout1_2;   // 应用 / 刷新
    QHBoxLayout *hlayout1_3;   // 导入 / 导出 / 隐藏详情
    QListWidget *layerList;
    QPushButton *addLayerBtn;
    QPushButton *removeLayerBtn;
    QPushButton *applyBtn;
    QPushButton *refreshBtn;
    QPushButton *importBtn;
    QPushButton *exportBtn;
    QPushButton *hideBtn;      // 折叠/展开 widget2（放 widget1，因 widget2 整个隐藏后内部按钮够不到）

    QWidget *widget2;
    QVBoxLayout *vlayout2_1;
    QWidget *detailTop;
    QHBoxLayout *hlayout2_top;
    QLineEdit *nameEdit;
    QComboBox *typeCombo;
    QStackedWidget *detailStack;
    // 各类型页
    QWidget *inputPage, *convPage, *poolPage, *linearPage, *dropoutPage, *reluPage, *flattenPage;
    QSpinBox *inputC, *inputH, *inputW;
    QSpinBox *convOut, *convKernel, *convStride, *convPadding, *convDilation;
    QSpinBox *poolKernel, *poolStride, *poolPadding;
    QSpinBox *linearOut;
    QDoubleSpinBox *dropoutP;

    QWidget *widget3;
    QVBoxLayout *vlayout3_1;
    QSplitter *vsplitter3;
    QWidget *trainPanel;
    QVBoxLayout *trainLayout;
    QComboBox *formatCombo;
    QPushButton *loadDatasetBtn;
    QLabel *datasetLabel;
    QPushButton *randomDataBtn;
    QComboBox *deviceCombo;
    QSpinBox *batchSpin;
    QDoubleSpinBox *lrSpin;
    QSpinBox *epochSpin;
    QPushButton *startBtn;
    QPushButton *pauseBtn;
    QPushButton *abortBtn;
    QPushButton *saveWeightsBtn;
    QPushButton *loadWeightsBtn;
    Plot_Widget *lossPlot;

    // —— 数据 / 模型 / 训练 ——
    QJsonObject json_;            // 编辑期唯一真相源
    Model model_;                 // 唯一实例，训练前/暂停后改结构
    Dataset dataset_;             // 已加载的数据
    TrainingWorker *worker = nullptr;
    QThread *thread = nullptr;
    bool modelApplied_ = false;   // 「应用」成功后置 true，训练/存权重前检查
    bool updatingDetail_ = false; // 回填详情页时抑制 valueChanged 写回

    // —— 帮助函数 ——
    void initDefaultJson();
    void rebuildLayerList(int selectRow = -1);
    void loadDetailPage();
    int currentLayerIndex() const;               // 列表行 → 层数组下标，-1=输入行
    QJsonObject currentLayerJson() const;
    void setLayerField(const QString& key, QJsonValue v);
    void setInputShapeField(int idx, int v);     // idx 0/1/2 = C/H/W
    void showLayerPage(const QString& type);     // 切 stack 页
    QJsonObject defaultLayerForType(const QString& type);
    void applyJsonToModel();                     // 包 try/catch 的「应用」
    bool ensureDatasetReady();
    bool ensureModelReady();
    torch::Device currentDevice() const;
    void setTrainingUI(bool training);
    QString currentInputShapeText() const;       // "C×H×W"

    // —— 事件处理（普通成员函数，指针 connect 直接用）——
    void onAddLayer();
    void onRemoveLayer();
    void onApply();
    void onRefresh();
    void onImportJson();
    void onExportJson();
    void onLayerSelected(int row);
    void onNameChanged(const QString& text);
    void onTypeChanged(int idx);
    void onToggleWidget2();
    void onLoadDataset();
    void onRandomData();
    void onDeviceChanged(int idx);
    void onLrChanged(double v);
    void onStart();
    void onPause();
    void onAbort();
    void onSaveWeights();
    void onLoadWeights();
    void onLoss(float loss);
    void onEpochDone(int epoch);
    void onTrainingFinished();

    QString randomDir_;   // 随机数据临时目录
    AugConfig aug_;

    int trainedEpochs_ = 0;

protected:
    void closeEvent(QCloseEvent *event);
};

#endif // MAINWINDOW_H
