/* 
 * File:   scalar.cpp
 * Author: renato
 * 
 * Created on November 16, 2014, 3:21 AM
 */

#include "scalar.h"
#include <float.h>
#include <stdexcept> 

Scalar::Scalar(std::string _name) {
    name = _name;
    min = FLT_MAX;
    max = -FLT_MAX;
}

Scalar::~Scalar() {}

std::vector<float>& Scalar::getValues() {
    return values;
}

void Scalar::setValues(std::vector<float>& _values) {
    values = _values;
    
    for (int i = 0; i < _values.size(); i++) {
        if (_values[i] > max)
            max = _values[i];
        if (_values[i] < min)
            min = _values[i];
    }
    
}

void Scalar::setValue(int index, float value) {
    if (values.size() > index) {
        values[index] = value;
        if (value > max)
            max = value;
        if (value < min)
            min = value;
    }
    else
        throw std::out_of_range ("Index out of range!");
         
};

float Scalar::getValue(int index) {
    if (values.size() > index)
        return values[index];
    else
        throw std::out_of_range ("Index out of range!");
}

float Scalar::getMin() {
    return min;
}

float Scalar::getMax() {
    return max;
}
   
std::string Scalar::getName() {
    return name;
};


