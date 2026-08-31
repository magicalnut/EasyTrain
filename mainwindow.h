#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#pragma once

#include "monitor.h"

#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QCloseEvent>
#include <QSplitter>

// Qt 的 qobjectdefs.h 把 `slots` 定义成空宏（#define slots），会和 libtorch 的
// Object::slots() 撞名：ivalue_inl.h:1585 的 `...& slots() const {` 被预处理成
// `...& () const {` → C2059/C2334。所有 Qt 头 include 完先 undef，再引入 torch。
// 注意不能 undef `signals`：本文件下面还在用 signals:。
#undef slots

class MainWindow : public QWidget
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
private:
    Monitor *monitor;

    QPushButton *button1;

    QVBoxLayout *mainlayout;
    QSplitter *hsplitter1;

    QWidget *widget1;
    QVBoxLayout *vlayout1_1;
    QHBoxLayout *hlayout1_1;
    QPushButton *button1_1;
    QPushButton *button1_2;

    QWidget *widget2;

    QWidget *widget3;
    QVBoxLayout *vlayout3_1;

protected:
    void closeEvent(QCloseEvent *event);
};

#endif // MAINWINDOW_H
