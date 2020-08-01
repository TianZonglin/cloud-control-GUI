/* 
 * File:   explanatorymap.cpp
 * Author: renato
 * 
 * Created on April 8, 2016, 1:29 AM
 */

#include "explanatorymap.h"
#include "include/projutil.h"

ExplanatoryMap::ExplanatoryMap() {
}

ExplanatoryMap::ExplanatoryMap(const ExplanatoryMap& orig) {
}

ExplanatoryMap::~ExplanatoryMap() {
}
 


void ExplanatoryMap::buildTopRankedNew() {

cout<<"\n"<<"----------------"<<__PRETTY_FUNCTION__<<endl;


    int currentCmapBak = current_cmap;
    int invert_colormapBak = invert_colormap;
    invert_colormap = 0;
    colorMap.load(CMAP_CATEGORICAL);

    int NP = cloud->points.size();
    int numColors;
    if (cloud->attributes.size() > colorMap.getSize() - 1)
        numColors = colorMap.getSize() - 1;
    else
        numColors = cloud->attributes.size();

    //1 - Create a vector of scalar values for the projected points
    vector<float> dimColorIndex(NP, -1.0f);
    vector<float> dimContributions(NP, 0.0f);
    
    //2 - Get the top-ranked (# of colors) dimensions
    vector<int>& topDims = cloud->dimrank_topdims;

    //3 - Fill the scalar vector based on the available colors
    //    float minContribution = 1.0f;
    //    float maxContribution = 0.0f;

    
    for (int i = 0; i < NP; i++) {
        
        int dimId = cloud->point_dimrank_visual[i].dimId;
        vector<int>::iterator it = std::find(topDims.begin(), topDims.end(), dimId);
        if (it == topDims.end()){
            //(".");//全在这里输出,说明topDims是空的
            dimColorIndex[i] = colorMap.getSize()-1;    
        }   
        else{
            //printf("O");//
            dimColorIndex[i] =  std::min(int(it - topDims.begin()), numColors);
        }
            
        //printf("%f ",dimColorIndex[i]);//8.00000
        dimContributions[i] = cloud->point_dimrank_visual[i].weight;                
    }
    
       
    //From here on we expect that every observation has an assigned scalar (color), 
    //and a respective weight (brightness)
    Scalar scalarDimensions("Dimension Ranking");
    scalarDimensions.setValues(dimColorIndex);
    Scalar scalarContributions("Contributions");
    scalarContributions.setValues(dimContributions);
    
    //The scalar mapping follows the colormap indices: 0 for purple, 1 for gold, 2 for dark red and so on...
    const vector<float>& visualDimIds = scalarDimensions.getValues();
    const vector<float>& visualDimWeights = scalarContributions.getValues();
     
    
    //Create scalars to be used on the computation of outlines
    cloud->point_visualrank_scalar.clear();
    for (int i = 0 ; i < cloud->points.size(); i++) {
        cloud->point_visualrank_scalar.push_back(DimensionRank(visualDimIds[i], visualDimWeights[i]));
    }
    
    //5 - Generate the texture

    ////if (shepard_averaging <= 0.001f)
        image = buildWithoutShepard(scalarDimensions, scalarContributions);




    ////else
    ////    image  = ImageInterpolator::shepardCategorical(*cloud, scalarDimensions, 
    ////                 scalarContributions, shepard_averaging, dimrankUseBrightness);
    
    colorMap.load(currentCmapBak);
    invert_colormap = invert_colormapBak;    
}




float* ExplanatoryMap::buildWithoutShepard(Scalar &dimensions, Scalar &contributions) {
    
    cout<<"\n--------------------"<<__PRETTY_FUNCTION__<<endl;
    image = new float[canvasSize*canvasSize*4];
    vector<float> scalarValues = dimensions.getValues();
    //rad_max: Image is empty further away from the points than the cloud's average-dist..
    float rad_max  = cloud->avgdist;
    int   mapSize  = cloud->fboSize;
    vector<int> res;
    int closestPointId;
    vector<float> pointsContribution = contributions.getValues();
    
    cout<<"\n------------------------Loop searchNN()"<<endl;
    cout<<"\n------------------------Loop rgb2hsv & hsv2rgb"<<endl;
    for(int i=0,offs=0;i<mapSize;++i) {
        for(int j=0;j<mapSize;++j,++offs) {
            //The current pixel to shade
            const Point2d p(j,i);
            //Distance of current pixel to closest cloud point
            float dt = cloud->siteDT[offs];
            //Too far away from the cloud? Nothing to draw, no points there
            if (dt>rad_max) {
                //Set alpha to transparent
                image[4*offs+3] = 0;
                continue;
            }
        //printf("%f, %f\n",p.x, p.y);//正常的
            cloud->searchNN(p,1,res);	
            closestPointId = res[0];
        //printf("%d\n",closestPointId);//正常的
            float pointScalarVal = scalarValues[closestPointId];
            
        

            Color pointColor = colorMap.getColor(pointScalarVal);





            float r = pointColor.r, g = pointColor.g, b = pointColor.b;
        //printf("%f, %f, %f\n",r,g,b); //////get the same color WRONG!!!!!!!!!!!!!!!!!!!


            ///if (useDimRankingSaturation && !cloud->point_dimrank[closestPointId].empty()) {
            if (dimrankUseBrightness) {
                float h, s, v;
                rgb2hsv(r, g, b, h, s, v);
                if (contributions.getMax() - contributions.getMin() < 0.00000f)
                    v = pointsContribution[closestPointId] / contributions.getMax();
                else
                    v = (pointsContribution[closestPointId] - contributions.getMin()) / (contributions.getMax() - contributions.getMin());
                v = std::pow(v, dimRankBrightnessControl);
                
                hsv2rgb(h, s, v, r, g, b);
            }
                     
            //Generate the final color image:
            image[4*offs+0] = r;
            image[4*offs+1] = g;
            image[4*offs+2] = b;
            float alpha = 1-dt/rad_max;	
            image[4*offs+3] = alpha;				//Alpha encodes the distance to cloud
	    }
    }
    //for(int i=0;i<canvasSize;i=i+4) {
	//	printf("%f, %f, %f, %f\n",image[i+0],image[i+1],image[i+2],image[i+3]);
	//}

    return image;
}

















