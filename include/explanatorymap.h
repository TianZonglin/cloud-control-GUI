/* 
 * File:   explanatorymap.h
 * Author: renato
 *
 * Created on April 8, 2016, 1:29 AM
 */

#ifndef EXPLANATORYMAP_H
#define	EXPLANATORYMAP_H

#include "include/pointcloud.h"
#include "include/io.h"
#include "include/scalarimage.h"
#include "include/config.h"
#include "include/scalar.h"

class ExplanatoryMap {
public:
    ExplanatoryMap();
    ExplanatoryMap(const ExplanatoryMap& orig);
    virtual ~ExplanatoryMap();
    
    void buildTopRanked();
    void buildTopRankedNew();
    void buildTopRankedSet();
    
    float* image;
    PointCloud *cloud;
    
private:

    float* buildWithoutShepard(Scalar &dimensions, Scalar &contributions);
    
    
};

#endif	/* EXPLANATORYMAP_H */

