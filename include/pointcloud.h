#pragma once

#include <map>
#include <set>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include "ANN/ANN.h"
#include "include/hashwrap.h"
#include "include/orderedmatrix.h"
#include "include/point2d.h"
#include "grouping.h"

typedef ANNkd_tree ANNtree;		//type of ANN tree to use; can be ANNkd_tree or ANNbd_tree

class  FullMatrix;
class  SparseMatrix;
class  SortedErrorMatrix;
class  Grouping;


////struct Color					//General-purpose color	
////{
////	float r,g,b;
////		  Color() { r=g=b=1; }
////		  Color(float r_,float g_,float b_): r(r_),g(g_),b(b_) {}
////	Color operator*(float t) const { return Color(r*t,g*t,b*t); }
////	Color operator+(const Color& rhs) const { return Color(r+rhs.r,g+rhs.g,b+rhs.b); }
////};


struct Triangle					//Triangle defined by 3 point-ids in a given cloud
{
		Triangle() {}
		Triangle(int a,int b,int c) { idx[0]=a; idx[1]=b; idx[2]=c; }
	int operator()(int i) const { return idx[i]; }	

	int idx[3];
};
	


struct DimensionRank {    
    int dimId;                                                                  //Id (index) of the dimension
    float weight;                                                               //Weight, compared to the other dimensions
        
    DimensionRank() {
        dimId = 0; 
        weight = 0.0f; 
    }
    
    DimensionRank(int _dimId, float _weight) { 
        dimId = _dimId;
        weight = _weight;
    }
    
    DimensionRank(const DimensionRank& orig) {this->dimId = orig.dimId; this->weight = orig.weight;}
    
    DimensionRank& operator=(const DimensionRank& other)
    {  dimId = other.dimId; weight = other.weight; return *this; }
    
    struct SortByIdAsc
    {
        bool operator()( const DimensionRank& lx, const DimensionRank& rx ) const {
    	    return lx.dimId < rx.dimId;
        }
    };
  
    struct SortByWeightDesc
    {
        bool operator()( const DimensionRank& lx, const DimensionRank& rx ) const {
    	    return lx.weight > rx.weight;
        }
    };

    struct SortByWeightAsc
    {
        bool operator()( const DimensionRank& lx, const DimensionRank& rx ) const {
    	    return lx.weight < rx.weight;
        }
    };

    
};


struct DimensionHistogram {
    int dimId;
    int frequency;
    
    DimensionHistogram() {
        dimId = 0;
        frequency = 0;
    }
    
    DimensionHistogram(int _dimId, int _frequency) {
        dimId = _dimId;
        frequency = _frequency;
    }
    
    struct SortByIdAsc
    {
        bool operator()( const DimensionHistogram& lx, const DimensionHistogram& rx ) const {
    	    return lx.dimId < rx.dimId;
        }
    };
    
    struct SortByFrequencyAsc
    {
        bool operator()( const DimensionHistogram& lx, const DimensionHistogram& rx ) const {
    	    return lx.frequency < rx.frequency;
        }
    };

    struct SortByFrequencyDesc
    {
        bool operator()( const DimensionHistogram& lx, const DimensionHistogram& rx ) const {
    	    return lx.frequency > rx.frequency;
        }
    };
    
};



typedef std::vector<Point2d> PointSet;
typedef std::vector<DimensionRank> DimRankingList;

//----------------------------------------------------------------



class PointCloud							//2D point cloud, with many related helpers (knn, DT, FT, Delaunay, ...)
{
public:

struct Edge {								//An edge formed by 2 points in the cloud. The edge always belongs to a point (its 1st point), not stored in the edge 
				Edge(): pid(0),angle(0) {}
				Edge(int pid_,float angle_):pid(pid_),angle(angle_) {}
		float	deg() const { return 360*angle/2/M_PI; }			
				int pid;					//The 2nd edge point	
				float angle;				//Angle of the edge [0..2*M_PI] with +x axis
			};

class EdgeMatrix : public OrderedMatrix<Edge>										//For each point in the cloud, lists all its edges (to other NEAR points), sorted counterclockwise	
{
public:
						EdgeMatrix(int nrows): OrderedMatrix<Edge>(nrows) {}
};			

typedef hash_set<int>	TrisOfPoint;												//Idxs of all triangles that contain a given point in the cloud
typedef FullMatrix		DistMatrix;													//Symmetric distance matrix between all points in the cloud


						PointCloud(int fboSize);
					   ~PointCloud();
bool					loadPex(const char* filebase,const char* projname,bool load_nd);	//Load a set of files (nD data, 2D projection, proj-error) from PEx


bool myLoadPex(const char* filebase,const char* projname,bool load_nd);	//Load a set of files (nD data, 2D projection, proj-error) from PEx
void myFitToSize(float minX = 1.0e+8, float minY = 1.0e+8, float maxX = -1.0e+8, float maxY = -1.0e+8);
//reduce the dimensionality, keeping dimensions high higher variance
void myReduceAttributes(int toDimensionaloty);
void myReduceAttributes(std::vector<int> selectedDimsIds);   
void dimensionRankCentroid();
DimRankingList centroidRanks;
DimRankingList dimensionRankContribution(int pid, std::vector<int> &neighbors, int strategy);
        


        //Compute the most important dimension to a point pid, given its neighborhood
        DimRankingList dimensionRank(int pid, float radius);
        //Computer dimension importance of all projected points, given a radius t
        void filterRankings(bool topdims_only);
        void dimensionRank(float radius);
        void computeTopDims();
        //compute the cloud diameter
        void computeDiameter();

        //std::vector<std::vector<float>*> attributes;
        std::vector<float> attributes_mean;
        //std::vector<float> attributes_min;
        //std::vector<float> attributes_max;
        std::vector<std::string> attributes_names;                                                                              
        std::vector<double> attributes_variance;                                //Stores variance of each attribute of the nd-matrix
        
        std::vector<int> attributes_ids;                                        //Stores attrs ids
        std::map<int, int> attributes_indices;                                  //<dimId, position on attributes_ids> Stores in which positions of attributes_ids each id appears
        
        std::vector<std::vector<float>*> attributes_original;
        std::vector<float> attributes_original_mean;
        std::vector<float> attributes_original_min;
        std::vector<float> attributes_original_max;
        std::vector<std::string> attributes_original_names;
        //Stores variance of each attribute of the nd-matrix 
        std::vector<double> attributes_original_variance;                                       //<dimId, position on attributes_ids> Stores in which positions of attributes_ids each id appears
          
		//Attributes importance (used in Multidimensional Ranking Map)
        //For each point, store the dimension ranking of its neighborhood
        std::vector<DimRankingList> point_dimrank;
        std::vector<DimensionRank> point_dimrank_visual;
        //Store the visual ranking confidence and scalar (to convert into a color code)
        DimRankingList point_visualrank_scalar;   


		float diameter;

        std::vector<int> dimrank_topdims;
        std::vector<DimRankingList> stored_dimrank;

        //Store the frequency of importance per point/dimension, given a ranking metric
        std::vector<DimensionHistogram> dimHistogram;


void					initEnd();
int						size() const { return points.size(); }
int						numLabels() const { return num_labels; }
int						dimensions() const { return attributes.size(); }
int						searchNN(const Point2d& seed,int k,std::vector<int>& result,std::vector<float>* result_d=0) const;
int						searchR(const Point2d& seed,float rad,int nn_max,std::vector<int>& result) const;
int						closest(int pid,float& d) const;							//Return point-id and distance to closest point to 'pid' in cloud
void					closestEdges(const Point2d& x, int pid,const Edge*& e1,float& d1,const Edge*& e2,float& d2) const;
float					interpolateDistMatrix(const Point2d& pix,float& certainty,float delta) const;
float					blendDistance(const Point2d& pix,const Triangle&) const;
int						hitTriangle(const Point2d& x) const;						//Return triangle (pid,e1,e2) that contains point x; false if no such triangle exists
bool					findTriangle(const Point2d& x,int& pid,const Edge*& e1,const Edge*& e2) const;
void					sortErrors();
void					computeFalseNegatives(int pid,bool norm=false);				//Compute false-negative error (relative_error[]) w.r.t. pid
void					computeFalseNegatives(const Grouping::PointGroup&,float range);			
																					//Compute false-negative error (relative_error[]) w.r.t. entire given group
void					computeAggregateError(float norm);							//Compute aggregate_error[] 
void					computeLabelMixing();										//Compute mixing of labels (in [0,1]) around each point
void					triangulate();												//Compute the exact Delaunay triangulation of the cloud
Grouping*				groupByLabel();												//Construct a grouping of this based on the (int) value of point-scalars
float					averageNeighborDist(int pid) const;							//Return average dist to geometric nbs of point 'pid'
	
std::vector<Point2d>	points;														//The cloud points
Point2d					min_p,max_p;												//Bounding box for points[]
std::vector<float>		label_mix;													//The degree of label-mixing at each point (in [0,1], 0=pure, 1=completely mixed)
std::vector<float>		point_scalars;												//Scalar data for points (can encode anything you want)
float					point_scalars_min,point_scalars_max;						//Range for point_scalars[]
ANNtree*				kdt;														//KDT for points
ANNpointArray			kdt_points;													//KDT for points (helper)
FullMatrix*				distmatrix = NULL;													//Distance matrix (or another error matrix) for all points	
//FullMatrix *errorMatrix = NULL;
FullMatrix *sqrDistanceMatrix = NULL; 
std::vector<std::string> point_names;
SparseMatrix*			edges;														//Edges of Delaunay triangulation of points: edges[i] contains all point-idxs that are connected to i via Delaunay edges
EdgeMatrix*				sorted_edges;												//Delaunay edges (as above), but sorted anticlockwise around each vertex. Useful for fast spatial point-between-edges search
SortedErrorMatrix*		sorted_errors;
std::vector<float>		false_negative_error;										//For a given point pid, all errors of all other points to pid
std::vector<float>		aggregate_error;											//Projection error, aggregated for a point w.r.t. all its neighbors
std::vector<float>		aggregate_fp_error;
std::vector<float>		aggregate_fn_error;
std::vector<Triangle>	triangles;													//Delaunay triangulation of the point set
std::vector<TrisOfPoint> point2tris;												//Triangles sharing each point
std::vector<std::vector<float>*> attributes;										//n-dimensional attributes of points
std::vector<float>		attributes_min;
std::vector<float>		attributes_max;

unsigned int*			buff_triangle_id;											//Per image-pixel, the triangle-id+1 (of the Delaunay triangle covering that pixel, if any), or 0 if no triangle there

float*					siteParam;													//Point parameterization (fboSize^2). siteParam(i,j) = point-id+1 if there's a point at (i,j), else 0
short*					siteFT;														//FT of points (fboSize^2)
float*					siteDT;														//DT of points (fboSize^2)
float					DT_max;														//Max value in siteDT[]
float					siteMax;
float					avgdist;													//Average inter-point distance in the cloud
int						fboSize;													//Size of various images used in here
int						num_labels;													//# different labels in the cloud

private:
void					makeKDT();
};
















//--------  Inlines  ----------------------------------------


inline int PointCloud::hitTriangle(const Point2d& p) const					//Return triangle-id that contains point x; returns -1 if no such triangle exists
{
	int tid = buff_triangle_id[int(p.x)+fboSize*int(p.y)];					//Index triangle-map to see what ID we have at that pixel
	if (!tid) return -1;													//Zero means no triangle
	return tid-1;															//Nonzero means triangle-id + 1
}


