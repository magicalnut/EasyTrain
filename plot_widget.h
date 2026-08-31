#ifndef PLOT_WIDGET_H
#define PLOT_WIDGET_H
#pragma once

#include <QPainter>
#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

class Plot_Widget : public QWidget
{
    Q_OBJECT
public:
    Plot_Widget(int times, QWidget *parent = nullptr, double ymax = 100, double ymin = 0);
    ~Plot_Widget();
    void add_point(double p);
private:
    // QVBoxLayout *layout;
    // cv::Mat image;
    int maxx;
    double maxy;
    double miny;
    double maxy_floor;
    double miny_floor;
    std::list<double>*y;
    double itemx;
    double itemy;
    // QLabel *label;
protected:
    void paintEvent(QPaintEvent *event);
};

#endif // PLOT_WIDGET_H
