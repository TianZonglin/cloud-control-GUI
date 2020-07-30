/* 
 * File:   scalar.h
 * Author: renato
 *
 * Created on November 16, 2014, 3:21 AM
 */

#ifndef SCALAR_H
#define	SCALAR_H

#include <iostream>
#include <vector>

class Scalar {
public:
    Scalar(std::string _name);
    virtual ~Scalar();
    
    std::vector<float>& getValues();
    void setValues(std::vector<float>& _values);
    void setValue(int index, float value);
    float getValue(int index);
    float getMin();
    float getMax();
    std::string getName();
    
private:
    std::string name;
    std::vector<float> values;
    float min, max;    
};

#endif	/* SCALAR_H */

