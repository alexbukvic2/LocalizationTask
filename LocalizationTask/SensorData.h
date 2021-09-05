#ifndef SENSORDATA_H
#define SENSORDATA_H
#include<Eigen/Dense>

using namespace Eigen;

typedef Matrix<double, 2, 1> Position;
struct SensorData
{
    int timestampMs;
    Position position;
    double distance;
};

#endif