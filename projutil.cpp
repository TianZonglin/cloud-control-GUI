/* 
 * File:   projutil.cpp
 * Author: renato
 * 
 * Created on June 24, 2014, 3:28 PM
 */

#include "include/projutil.h"
#include <utility>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <fstream> 



ProjUtil::ProjUtil() {
}

ProjUtil::~ProjUtil() {
}

MatrixXf ProjUtil::covariance(MatrixXf &mat) {
    int rows = mat.rows();
    int cols = mat.cols();
    
    MatrixXf covMatrix(cols, cols);
    std::vector<float> means;
    
    //1 - computing cols means, and storing in means vector
    for (int j = 0; j < cols; j++) {
        float meanJ = 0.0f;
        for (int i = 0; i < rows; i++) {
            meanJ += mat(i, j);            
        }
        meanJ /= rows;
        means.push_back(meanJ);
    }
    
    //2 - computing covariance for each pair of dimensions    
    for (int i = 0; i < cols; i++) {
        for (int j = i; j < cols; j++) {
            float meanA = means[i];
            float meanB = means[j];
            float cov = 0.0f;
            for (int line = 0; line < rows; line++)
                cov +=  (mat(line, i) - meanA) * (mat(line, j) - meanB);            
            cov /= (rows-1);
            covMatrix(i,j) = cov;
            covMatrix(j,i) = cov;
       }
    }
    return covMatrix;
}


/**
 * Transforms a string in a integer value (a hash code).
 * @param str The original string.
 * @return Its integer code.
 */
int ProjUtil::hashCode(string str) {
    
    if (str.length() == 0)
        return -1;
            
    //Trim this string (remove \n\r\t and whitespaces before and after)
    str = trim(str);
    
    //convert all characters of the id to lowercase
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    
    int h = 0;
    for (int i = 0; i < str.length(); i++) {
        h = 31 * h + str[i];
    }

    return h;        
}

/**
 * Removes any character \n\r\t and whitespaces before and after this string
 * @param str String to be trimmed
 * @return Trimmed string
 */
string ProjUtil::trim(string str) {
    
    str.erase(str.find_last_not_of(" \n\r\t")+1);
    if (!str.empty()) 
        str = str.substr(str.find_first_not_of(" \n\r\t"));
    
    return str;
}

/**
 * Checks whether the argument string is a number.
 * @param str The original string.
 * @return True is it is a number, False otherwise.
 */
bool ProjUtil::isNumber(string str) {
   if(str.empty() || ((!isdigit(str[0])) && (str[0] != '-') && (str[0] != '+'))) 
       return false ;

   char * p ;
   strtol(str.c_str(), &p, 10) ;

   return (*p == 0) ;
}


vector<string> ProjUtil::explode(const string & str, char delim)
{
    vector<string> elems;
    stringstream ss(str);
    string item;
    
    while (std::getline(ss, item, delim)) {
        item = ProjUtil::trim(item);
        if (item.length() > 0)
            elems.push_back(item);
    }
    
    return elems;
}

double ProjUtil::variance(const vector<double>& v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double m =  sum / v.size();

    double accum = 0.0;
    for (int i = 0; i < v.size(); i++) {
        accum += (v[i] - m) * (v[i] - m);
    };
  
    return accum / (v.size()-1); 
}

string ProjUtil::toString(const float& t) { 
   std::ostringstream os; 
   os<<t;
   std::string s(os.str());
   return s; 
}

bool ProjUtil::fileExists(string filename) {
    ifstream ifile(filename.c_str());
    return (bool)ifile;    
}
