/* 
 * File:   projutil.h
 * Author: Renato Rodrigues Oliveira da Silva (rros@icmc.usp.br)
 *
 * Created on June 24, 2014, 3:28 PM
 */

#ifndef PROJUTIL_H
#define	PROJUTIL_H

#include <iostream>
#include <vector>
//#include <Eigen/Eigen>
//#include <Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

template <class T1, class T2, class Pred = std::less<T2> >
struct sort_pair_second {
    bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
        Pred p;
        return p(left.second, right.second);
    }
};

template <class T1, class T2, class Pred = std::greater<T2> >
struct sort_pair_second_desc {
    bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
        Pred p;
        return p(left.second, right.second);
    }
};


template <class T1, class T2, class Pred = std::less<T2> >
struct sort_pair_first {
    bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
        Pred p;
        return p(left.first, right.first);
    }
};


class ProjUtil {
    
public:
    ProjUtil();
    virtual ~ProjUtil();
    
    static int hashCode(string str);
    static bool isNumber(string str);
    static bool fileExists(string filename);
    static string trim(string str);
    static vector<string> explode(const string &str, char delim);
    static double variance(const vector<double>& v);
    static double distance(vector<float>* v1, vector<float>* v2);
    static string toString(const float& value);
    static MatrixXf covariance(MatrixXf &mat);
private:
    
};

#endif	/* UTIL_H */