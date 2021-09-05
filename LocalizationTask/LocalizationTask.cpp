#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <boost/tokenizer.hpp>
#include "SensorData.h"
#include <math.h>
#include <Eigen/src/Core/Matrix.h>

using namespace std;
using namespace boost;

typedef tokenizer<escaped_list_separator<char>> Tokenizer;

const int OBJECTS_COUNT = 2;
const int SENSORS_COUNT = 3;

Position sensorPos[SENSORS_COUNT];

vector<string> readCsvLine(string line)
{
    vector<string> vec;
    Tokenizer tok(line);
    vec.assign(tok.begin(), tok.end());
    return vec;
}

Matrix<double, 2 * OBJECTS_COUNT, 1> readInitialPositions(string line)
{
    vector<string> vec = readCsvLine(line);
    Matrix<double, 2 * OBJECTS_COUNT, 1> initPos;
    initPos << stof(vec[0]), stof(vec[1]),
        stof(vec[2]), stof(vec[3]);
    return initPos;
}

SensorData readSensorData(string line) {
    Tokenizer tok(line);
    vector<string> vec = readCsvLine(line);

    int timestamp = stoi(vec[0]);
    Position position;
    position << stof(vec[1]), stof(vec[2]);
    float distance = stof(vec[3]);

    return { timestamp, position, distance };
}

double calculateDistance(Position pos1, Position pos2)
{
    return sqrt(pow(pos1(0, 0) - pos2(0, 0), 2) + pow(pos1(1, 0) - pos2(1, 0), 2));
}

Matrix<double, SENSORS_COUNT, OBJECTS_COUNT> estimateDistances(Matrix<double, 2 * OBJECTS_COUNT, 1> objPosEst)
{
    // based on position estimation, estimate distances to each sensor
    Matrix<double, SENSORS_COUNT, OBJECTS_COUNT> objSensorDistEst;
    for (int i = 0; i < SENSORS_COUNT; i++)
    {
        for (int j = 0; j < OBJECTS_COUNT; j++) {
            objSensorDistEst(i, j) = calculateDistance({ objPosEst(j * 2, 0), objPosEst(j * 2 + 1, 0) }, sensorPos[i]);
        }
    }
    return objSensorDistEst;
}

Matrix<double, SENSORS_COUNT, OBJECTS_COUNT> assignMeasurements(vector<SensorData> sensorData, Matrix<double, 
    SENSORS_COUNT, OBJECTS_COUNT> objSensorDistEst)
{
    // since it is not known which measurement comes for which object, using distance estimation we will assign corresponding measurements
    Matrix<double, SENSORS_COUNT, OBJECTS_COUNT> objSensorDist = MatrixXd::Zero(SENSORS_COUNT, OBJECTS_COUNT);
    int dataCount = sensorData.size();
    int i = 0;
    while (dataCount - 1 - i >= 0 && sensorData[dataCount - 1 - i].timestampMs == sensorData[dataCount - 1].timestampMs)
    {
        SensorData sensorDataAct = sensorData[dataCount - 1 - i];

        for (int iSens = 0; iSens < SENSORS_COUNT; iSens++)
        {
            if (sensorDataAct.position == sensorPos[iSens])
            {
                for (int iObj = 0; iObj < OBJECTS_COUNT; iObj++)
                {
                    if (objSensorDist(iSens, (iObj + 1) % OBJECTS_COUNT) != 0 ||
                        abs(objSensorDistEst(iSens, iObj) - sensorDataAct.distance) <
                        abs(objSensorDistEst(iSens, (iObj + 1) % OBJECTS_COUNT) - sensorDataAct.distance))
                    {
                        objSensorDist(iSens, iObj) = sensorDataAct.distance;
                        break;
                    }
                }
            }
        }
        i++;
    }
    return objSensorDist;
}

Matrix<double, 2 * OBJECTS_COUNT, 1> distancesToPositions(Matrix<double, SENSORS_COUNT, OBJECTS_COUNT> objSensorDist)
{
    // using trilateration + least squares method, convert distances from object to sensors to object position
    Matrix<double, 2 * OBJECTS_COUNT, 2 * OBJECTS_COUNT> A;
    A << 2 * (sensorPos[2](0) - sensorPos[0](0)), 2 * (sensorPos[2](1) - sensorPos[0](1)), 0, 0,
        2 * (sensorPos[2](0) - sensorPos[1](0)), 2 * (sensorPos[2](1) - sensorPos[1](1)), 0, 0,
        0, 0, 2 * (sensorPos[2](0) - sensorPos[0](0)), 2 * (sensorPos[2](1) - sensorPos[0](1)),
        0, 0, 2 * (sensorPos[2](0) - sensorPos[1](0)), 2 * (sensorPos[2](1) - sensorPos[1](1));
    Matrix<double, 2 * OBJECTS_COUNT, 1> b;
    b(0, 0) = pow(objSensorDist(0, 0), 2) - pow(objSensorDist(2, 0), 2) - pow(sensorPos[0](0), 2) - pow(sensorPos[0](1), 2) +
        pow(sensorPos[2](0), 2) + pow(sensorPos[2](1), 2);
    b(1, 0) = pow(objSensorDist(1, 0), 2) - pow(objSensorDist(2, 0), 2) - pow(sensorPos[1](0), 2) - pow(sensorPos[1](1), 2) +
        pow(sensorPos[2](0), 2) + pow(sensorPos[2](1), 2);
    b(2, 0) = pow(objSensorDist(0, 1), 2) - pow(objSensorDist(2, 1), 2) - pow(sensorPos[0](0), 2) - pow(sensorPos[0](1), 2) +
        pow(sensorPos[2](0), 2) + pow(sensorPos[2](1), 2);
    b(3, 0) = pow(objSensorDist(1, 1), 2) - pow(objSensorDist(2, 1), 2) - pow(sensorPos[1](0), 2) - pow(sensorPos[1](1), 2) +
        pow(sensorPos[2](0), 2) + pow(sensorPos[2](1), 2);

    Matrix<double, 2 * OBJECTS_COUNT, 1> position;
    position = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    return position;
}

void initializeSensorPositions() 
{
    sensorPos[0] << 0, 0;
    sensorPos[1] << 5, 0;
    sensorPos[2] << 0, 5;
}

int main()
{
    using namespace std;
    using namespace boost;
    string data("dataset.csv");

    ifstream in(data.c_str());
    if (!in.is_open()) return 1;

    initializeSensorPositions();
    
    string line;
    getline(in, line);
    Matrix<double, 2 * OBJECTS_COUNT, 1> objPosPrev = readInitialPositions(line);

    Matrix<double, OBJECTS_COUNT * 2, OBJECTS_COUNT * 2> predErrorCov = MatrixXd::Zero(OBJECTS_COUNT * 2, OBJECTS_COUNT * 2);
    Matrix<double, OBJECTS_COUNT * 2, OBJECTS_COUNT * 2> measNoiseCov;
    double measNoise = 0.5;
    measNoiseCov = measNoise * MatrixXd::Identity(OBJECTS_COUNT * 2, OBJECTS_COUNT * 2);

    Matrix<double, OBJECTS_COUNT * 2, OBJECTS_COUNT * 2> procNoise;
    double procNoiseSigma = 0.8;
    procNoise = pow(procNoiseSigma, 2) * MatrixXd::Identity(OBJECTS_COUNT * 2, OBJECTS_COUNT * 2);

    vector<SensorData> sensorData;
    int timestampPrevMs = 0;
    Matrix<double, 2 * OBJECTS_COUNT, 1> velocityPrev = MatrixXd::Zero(2 * OBJECTS_COUNT, 1);
    getline(in, line);
    do
    {
        sensorData.push_back(readSensorData(line));
        getline(in, line);
        int dataCount = sensorData.size();

        // if new timestamp is detected, calculate positions for current timestamp
        if (line == "" || readSensorData(line).timestampMs != sensorData[dataCount - 1].timestampMs)
        {
            int timestampMs = sensorData[dataCount - 1].timestampMs;
            int deltaT = timestampMs - timestampPrevMs;

            // calculate position estimation
            Matrix<double, 2 * OBJECTS_COUNT, 1> objPosEst;
            objPosEst = objPosPrev + deltaT * velocityPrev;

            // calculate estimation of distances between objects and sensors (based on position estimation)
            Matrix<double, SENSORS_COUNT, OBJECTS_COUNT> objSensorDistEst = estimateDistances(objPosEst);

            // assign corresponding measurements (based on estimated distances)
            Matrix<double, SENSORS_COUNT, OBJECTS_COUNT> objSensorDist = assignMeasurements(sensorData, objSensorDistEst);

            // calculate predicted error covariance
            Matrix<double, OBJECTS_COUNT * 2, OBJECTS_COUNT * 2> procNoiseCov;
            procNoiseCov = deltaT * procNoise;
            predErrorCov = predErrorCov + procNoiseCov;

            // trilateration - calculate positions from distances (using least squares method)
            Matrix<double, 2 * OBJECTS_COUNT, 1> meas = distancesToPositions(objSensorDist);

            // calculate measurement residual
            Matrix<double, 2 * OBJECTS_COUNT, 1> measRes = meas - objPosEst;

            // calculate Kalman gain
            Matrix<double, 2 * OBJECTS_COUNT, 2 * OBJECTS_COUNT> kalmanGain;
            kalmanGain = predErrorCov * (measNoiseCov + predErrorCov).inverse();

            // calculate position - update state estimation
            Matrix<double, 2 * OBJECTS_COUNT, 1> objPos;
            objPos = objPosEst + kalmanGain * measRes;
            cout << "timestamp: " << timestampMs << "ms | object 1 | x: " << objPos(0, 0) << ", y: " << objPos(1, 0) << endl;
            cout << "timestamp: " << timestampMs << "ms | object 2 | x: " << objPos(2, 0) << ", y: " << objPos(3, 0) << endl;

            // update error covariance
            predErrorCov = (MatrixXd::Identity(4, 4) - kalmanGain) * predErrorCov;

            // save data for next position estimation
            if (deltaT != 0)
            {
                velocityPrev = (objPos - objPosPrev) / deltaT;
            }
            objPosPrev = objPos;
            timestampPrevMs = timestampMs;
        }
    } while (line != "");
}
