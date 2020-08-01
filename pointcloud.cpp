#include "include/pointcloud.h"
#include "include/skelft.h"
#include "include/sparsematrix.h"
#include "include/fullmatrix.h"
#include "include/sortederrmatrix.h"
#include "include/grouping.h"
#include "include/cudawrapper.h"
#include "include/projutil.h"
#include "include/scalar.h"
#include "include/io.h"
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <deque>
#include <bits/stl_numeric.h>
#include <set>


#define REAL double									//Needed for Triangle; note also that Triangle is a C API

#define TRILIBRARY
#define NO_TIMER
#define ANSI_DECLARATORS
#define VOID void

extern "C" {
#include "triangle.h"
#include "projutil.h"
#include "config.h"
}


using namespace std;

 
float Point2d::edgeAngle(const Point2d& p, const Point2d& q)
{
			const Point2d& r  = q-p;
			float n           = r.norm();
			if (n<1.0e-4) return -1;							//Cannot determine angle, edge is of zero length

			float cos_r       = r.x/n;
			float sin_r       = r.y/n;
			float alpha       = acos(cos_r);
			if (sin_r<0)  alpha = 2*M_PI - alpha;
			return alpha;
}

float Point2d::angle(const Point2d& r)
{
			float n           = r.norm();
			if (n<1.0e-4) return -1;							//Cannot determine angle, edge is of zero length

			float cos_r       = r.x/n;
			float sin_r       = r.y/n;
			float alpha       = acos(cos_r);
			if (sin_r<0)  alpha = 2*M_PI - alpha;
			return alpha;
}

Point2d Point2d::center(const Point2d& a,const Point2d& b,const Point2d& c)
{
	Point2d ab = b-a; ab.normalize();
	Point2d ac = c-a; ac.normalize();
	Point2d cb = b-c; cb.normalize();
	Point2d ca = a-c; ca.normalize();

	Point2d  A = (ab+ac); A.normalize();
	Point2d  C = (ca+cb); C.normalize();

	float t2 = (A.x*(c.y-a.y)-A.y*(c.x-a.x))/(A.y*C.x-A.x*C.y);

	Point2d r;
	r.x = C.x*t2 + c.x;
	r.y = C.y*t2 + c.y;

	return r;
}

//-------------------------------------------------------------------------------------------------------------------------


PointCloud::PointCloud(int size): kdt(0),kdt_points(0),siteMax(0),fboSize(size),DT_max(0),
								  point_scalars_min(0),point_scalars_max(0),
								  avgdist(0),num_labels(0)
{
	cudaMallocHost((void**)&siteFT,size*size*2*sizeof(short));
	cudaMallocHost((void**)&siteParam,size*size*sizeof(float));
	cudaMallocHost((void**)&siteDT,size*size*sizeof(float));

	buff_triangle_id = new unsigned int[fboSize*fboSize];
	    //attributes_reduced_dim = -1;
        diameter = 0.0f;
        edges = 0;
        sorted_edges = 0;
        distmatrix = 0;
        sorted_errors = 0;
}
 





PointCloud::~PointCloud()
{
	if (kdt) { delete kdt; kdt=0; }
	if (kdt_points) annDeallocPts(kdt_points);
	delete edges;
	delete sorted_edges;
	delete distmatrix;
	delete sorted_errors;
	for(int i=0;i<attributes.size();++i)
	   delete attributes[i];

	delete[] buff_triangle_id;

	cudaFreeHost(siteFT);
	cudaFreeHost(siteParam);
	cudaFreeHost(siteDT);
}

void PointCloud::makeKDT()
{
    cout<<"\n"<<"----------------"<<__PRETTY_FUNCTION__<<endl;
	delete kdt;														//Deallocate whatever we hold currently
	if (kdt_points) annDeallocPts(kdt_points);

	kdt_points = annAllocPts(points.size(), 2);						//Put all vertices in the ANN search structure

	for(int i=0;i<points.size();i++)								//Copy surf vertices in an ANN-compatible structure
	{
		kdt_points[i][0]=points[i].x;
		kdt_points[i][1]=points[i].y;
	}

	kdt = new ANNtree(kdt_points, points.size(), 2, 5);
}

int PointCloud::searchR(const Point2d& seed,float rad,int nn_max,vector<int>& result) const
{
	ANNcoord query_pt[2];
	static ANNidx   nn_idx[10000];

	query_pt[0] = seed.x;
	query_pt[1] = seed.y;

	int nn = kdt->annkFRSearch(query_pt,rad*rad,nn_max,nn_idx);				//Search up to nn_max points within rad from seed.

	if (nn>nn_max) nn=nn_max;												//If there are more than nn_max points in the ball, beware that
																			//we only returned nn_max of them.
	result.resize(nn);
	for(int i=0;i<nn;++i)
		result[i] = nn_idx[i];

	return nn;
}


int PointCloud::closest(int pid,float& d) const
{
	ANNcoord query_pt[2];
	ANNidx   nn_idx[2];
    ANNdist  nn_dist[2];

	query_pt[0] = points[pid].x;
	query_pt[1] = points[pid].y;

	kdt->annkSearch(query_pt,2,nn_idx,nn_dist);

	d = sqrt(nn_dist[1]);

	return (nn_idx[1]!=pid)? nn_idx[1]:nn_idx[0];
}

int PointCloud::searchNN(const Point2d& seed,int k,vector<int>& result,vector<float>* result_dist) const
{
    //pout("--------------------");
	ANNcoord query_pt[2];
	static ANNidx   nn_idx[10000];
    static ANNdist  nn_dist[10000];

	query_pt[0] = seed.x;
	query_pt[1] = seed.y;

	kdt->annkSearch(query_pt,k,nn_idx,nn_dist);

	result.resize(k);
	if (result_dist) result_dist->resize(k);
	for(int i=0;i<k;++i)
	{
		result[i] = nn_idx[i];
		if (result_dist) (*result_dist)[i] = nn_dist[i];
	}

	return k;
}

void PointCloud::triangulate()
{

   cout<<"\n"<<"----------------"<<__PRETTY_FUNCTION__<<endl;
   triangulateio ti,to;
   double pts[50000];													//In: 2D points to be triangulated
   int tris[50000];														//Out: triangles created by the 2D triangulation
   int tedges[600000];													//Out: edges of triangulation

   ti.pointlist				   = pts;									//Set up triangulation of projected points
   ti.pointmarkerlist          = 0;										//
   ti.numberofpointattributes  = 0;										//
   ti.numberofpoints		   = points.size();

  // printf("points.size()=%d",points.size());

   to.trianglelist             = tris;									//All we want are the triangles..
   to.edgelist				   = tedges;								//..and edges
   to.pointmarkerlist          = 0;										//

   for(int i=0;i<ti.numberofpoints;++i)
   {
     pts[2*i]   = points[i].x;
	 pts[2*i+1] = points[i].y;
   }

   ::triangulate((char*)"zePBNQYY",&ti,&to,0);				//Call Triangle-lib to do the triangulation of the projected skel-points
	////generated into param 'to'
	printf("\n----------------Call Triangle-lib, edges = %d, tris = %d\n",to.numberofedges,to.numberoftriangles);

   triangles.resize(to.numberoftriangles);								//Get triangles:
   point2tris.resize(points.size());									//In the same time, construct point2tris[]
   for(int i=0;i<to.numberoftriangles;++i)
   {
	  int t = 3*i;
	  int a = to.trianglelist[t];
	  int b = to.trianglelist[t+1];
	  int c = to.trianglelist[t+2];
      triangles[i] = Triangle(a,b,c);

	  point2tris[a].insert(i);
	  point2tris[b].insert(i);
	  point2tris[c].insert(i);
   }


   for(int i=0;i<to.numberofedges;++i)									//Get edges:
   {
	  int i1 = tedges[2*i], i2 = tedges[2*i+1];
	  /////actually tedges is same withto.edgelist(equaled)
	  //printf("i1=%d, i2=%d\n",i1,i2); 
	  (*edges)(i1,i2) = 1;
	  (*edges)(i2,i1) = 1;
   }
   printf("\n----------------SparseMatrix generated\n");
}




struct EdgeCompare													//Compares two edges vs their angles [0..M_PI] with the +x axis
{																	//Used to sort edges anticlockwise
		 EdgeCompare() {}
	bool operator() (const PointCloud::Edge& ei,const PointCloud::Edge& ej)
					{ return ei.angle < ej.angle; }
};

/*
void PointCloud::sortErrors()										//For each matrix row (point in cloud), sort dist-errors to all other points.
{																	//Like this, it's next easy to find out, for any point, which are its most
	int NP = points.size();											//evident false positives (large negative dist-errors) and false negatives
	sorted_errors = new SortedErrorMatrix(NP);						//(large positive dist-errors)

	for(int i=0;i<NP;++i)
	{
		const DistMatrix::Row& row  = (*distmatrix)(i);
		SortedErrorMatrix::Row&  srow = (*sorted_errors)(i);

		int idx = 0;
		for(DistMatrix::Row::const_iterator it=row.begin();it!=row.end();++it,++idx)
		{
			float val = *it;
			srow.insert(make_pair(val,idx));
			 
		}
		printf("point(%d) has items %d\n",i,idx);
		////point(2093) has items 2100
		////point(2094) has items 2100
		////point(2095) has items 2100
		////point(2096) has items 2100
		////point(2097) has items 2100
		////point(2098) has items 2100
		////point(2099) has items 2100
	}
	///// distmatrix->print();  correct
}*/
void PointCloud::sortErrors()
{   
    cout<<"\n"<<"----------------"<<__PRETTY_FUNCTION__<<endl;
    if (distmatrix == NULL) {
        sorted_errors = 0;
        return;
    }
    int NP = points.size();
    sorted_errors = new SortedErrorMatrix(NP);
	
    for(int i=0;i<NP;++i)
    {
        const DistMatrix::Row& row  = (*distmatrix)(i);
        SortedErrorMatrix::Row&  srow = (*sorted_errors)(i);

        int idx = 0;
        for(DistMatrix::Row::const_iterator it=row.begin();it!=row.end();++it,++idx)
        {
            float val = *it;
            srow.insert(make_pair(val,idx));
        }
		////printf("point(%d) has items %d\n",i,idx);
		////point(2093) has items 2100
		////point(2094) has items 2100
		////point(2095) has items 2100
		////point(2096) has items 2100
		////point(2097) has items 2100
		////point(2098) has items 2100
		////point(2099) has items 2100
	}
	/////distmatrix->print();  //correct
}
double variance(const vector<double>& v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double m =  sum / v.size();

    double accum = 0.0;
    for (int i = 0; i < v.size(); i++) {
        accum += (v[i] - m) * (v[i] - m);
    };
  
    return accum / (v.size()-1); 
}

void PointCloud::initEnd(){

    cout<<"\n"<<"--------"<<__PRETTY_FUNCTION__<<endl;

    int NP = points.size();

    memset(siteParam,0,fboSize*fboSize*sizeof(float));				
    //Site parameterization:
    //0: no site at current pixel
    //i: site i-1 at current pixel    
    for(int i=0;i<NP;++i)
    {
        const Point2d& p = points[i];
        if (isnan(p.x) || isnan(p.y)) 
            continue;
        siteParam[int(p.y)*fboSize + int(p.x)] = i+1;
    }

    siteMax = NP+1;

    //Compute FT of the sites
    skelft2DFT(siteFT,siteParam,0,0,fboSize,fboSize,fboSize);
    
    //Compute DT of the sites (from the resident FT)
    skelft2DDT(siteDT,0,0,fboSize,fboSize);
  
    //Allocate Delaunay adjacency matrix 
    edges = new SparseMatrix(NP);
    //Allocate ordered Delaunay adj. matrix (edges ordered by angle anticlockwise)
    sorted_edges = new EdgeMatrix(NP);
    //Allocate space for various other data vectors that we can build now
    false_negative_error.resize(NP);
    aggregate_error.resize(NP);
    label_mix.resize(NP);

    //Compute max of DT
    DT_max = 0;
    for(int i=0;i<fboSize*fboSize;++i)
    {
        float dt = siteDT[i];
        if (dt>DT_max) DT_max = dt;
    }
	printf("\n--------siteDT generated.\n");

    //Compute Delaunay triangulation of this
    cout << "\n--------Triangulating cloud..." << endl;
    triangulate();

    //Used to compare edges vs their angles
    EdgeCompare edge_comp;
    //Construct sorted_edges[]: Delaunay edges[], but sorted anticlockwise around each point
    for(int i=0;i<NP;++i)
    {
        const SparseMatrix& ed = *edges;						
        //This is what we want to sort
        const SparseMatrix::Row& row = ed(i);
        //This is where the sorted output goes
        EdgeMatrix::Row& orow = (*sorted_edges)(i);
        //This is the current vertex, for which we'll sort edges
        const Point2d& p = points[i];

        //Make room for the sorted output
        orow.resize(row.size());
        int j=0;
        for(SparseMatrix::Row::const_iterator it=row.begin();it!=row.end();++it)
        {
            //Find the angle [0..M_PI] of edge (p,points[it->first]) with the +x axis
            const int pid = it->first;
            float alpha = Point2d::edgeAngle(p,points[pid]);
            //Copy the unsorted data to the sorted output, prior to sort
            orow[j++] = Edge(pid,alpha);
        }
        //Sort orow[] using the anticlockwise逆时针 edge order 
        std::sort(orow.begin(),orow.end(),edge_comp);

    }
	


    
    makeKDT();
	 
	cout << "\n--------Making pointcloud KDT finished" << endl;

	////////kdt->Print(true,std::cout);
 

    hash_set<int> lbls;
    //Compute average distance points in the cloud
    avgdist = 0;													
    //For this, we estimate the avg radius of the NSZ-nb, with a small NSZ
    //(for some reason, NSZ=approx 5 gives good results)
    for(int i=0;i<NP;++i) {
        const int NSZ = 5;
        vector<int> nn; vector<float> nnd;
        searchNN(points[i],NSZ,nn,&nnd);
        float dist = sqrt(nnd[NSZ-1]);
        avgdist += dist;
        lbls.insert(point_scalars[i]);
    }
    avgdist /= NP;
    num_labels = lbls.size();
 
    sortErrors();
 


    // global dispersion of each dimension
    cout << "\n--------Computing global variance..." << endl;
    attributes_variance.resize(attributes.size());
    for (int i = 0; i < attributes.size(); ++i) {
        vector<double> v((*attributes[i]).begin(),(*attributes[i]).end());
        //Normalizing
        for (int j = 0; j < v.size(); j++)
            v[j] = (v[j] - attributes_min[i])/(attributes_max[i] - attributes_min[i]);

        attributes_variance[i] = variance(v);    
    }
    
    computeDiameter();
    cout << "\n--------Cloud diameter: " << diameter << endl;;
    
    cout << "\n--------Cloud bounding box: ";
    cout << max_p.dist(min_p) << endl;
    printf("\n--------initEnd finished\n");
}


void PointCloud::computeDiameter() {
    cout<<"\n"<<"----------------"<<__PRETTY_FUNCTION__<<endl;
    int NP = points.size();
    diameter = 0.0f;
    for (int i = 0; i < NP; i++) {
        for (int j = i; j < NP; j++) {
            float distance = points[i].dist(points[j]);
            if (distance > diameter)
                diameter = distance;
        }
    }

    
}

Grouping* PointCloud::groupByLabel()								//Construct grouping of points in this based on equal-label
{
    cout<<"\n"<<"------------"<<__PRETTY_FUNCTION__<<endl;
	SimpleGrouping* grp = new SimpleGrouping(this);					//Make new grouping based on point-labeling of this
	map<int,int> gkey2gidx;											//Maps the group-ids to 0-based ids

	int NG=0,NP=size();
	for(int i=0;i<NP;++i)											//Read all points:
	{
		int gkey = int(point_scalars[i]);							//See if we have a new group-id
		if (gkey2gidx.find(gkey)==gkey2gidx.end())					//If so, map it to a 0-based increasing integer
		{
		   gkey2gidx.insert(make_pair(gkey,NG)); ++NG;
		}
	}

	grp->resize(NG);												//We now know how many groups we have to make
	for(int i=0;i<NP;++i)											//Scan the points and add them to their right groups
	{
		int gkey = int(point_scalars[i]);
		int gid  = gkey2gidx.find(gkey)->second;
		grp->group(gid).insert(i);
	}

	return grp;
}



bool PointCloud::loadPex(const char* file,const char* proj,bool load_nd)	//Load triplet of Pex files: nD, 2D, error
{
	const float t = 0.04;											//Border between the points and image-border (needed for safe DT computations)

	string fnd = file; fnd += ".nd";								//n-dimensional points file
	string f2d = file;												//2-dimensional projection file
	if (proj) { f2d += "."; f2d += proj; }
	f2d += ".2d";

	string fer = file;												//error matrix (wrt projection of n-D points to 2-D points)
	if (proj) { fer += "."; fer += proj; }
	fer += ".err";
	char line[1024];

	printf("\n$$$ file=%s, proj=%s, loadnd=%d\n",f2d.c_str(),fer.c_str(),load_nd);
	
	FILE* fp = fopen(f2d.c_str(),"r");								//1. Read 2D projections:
	if (!fp)
	{
		cout<<"Error: Cannot open "<<f2d<<endl;
		return false;
	}

	/* ///////数据格式
	DY
	2100 	///NP
	2		///dim

	0;0.21437205;0.6015861;0.0
	1;0.2005839;0.6126034;0.0
	2;0.16344348;0.46330938;0.0
	3;0.1700877;0.6452591;0.0
	4;0.2121075;0.60117453;0.0
	5;0.19401884;0.5529055;0.0
	6;0.17362374;0.6167138;0.0
	7;0.08745773;0.5757179;0.0
	//// &p.x,  &p.y,  &point_scalars[i]
	*/
	fgets(line,1024,fp);											//Skip first line 'DY'
	int NP;
	fscanf(fp,"%d",&NP);											//Get #points in file
	int dim;
	fscanf(fp,"%d",&dim);											//Get point dimensions (should be 2)
	if (dim!=2)
	{
	   cout<<"Warning: 2D projection dimension="<<dim<<", expected 2"<<endl;
	}

	////vector<Point2D>
	points.resize(NP);
	////vector<float>
	point_scalars.resize(NP);
	point_scalars_min = min_p.x = min_p.y = 1.0e+8;
	point_scalars_max = max_p.x = max_p.y = -1.0e+8;

	for(int i=0;i<NP;++i)											//Read all 2D point projections:
	{
		Point2d& p = points[i];
		fscanf(fp,"%*[^;];%f;%f;%f",&p.x,&p.y,&point_scalars[i]);	//REMARK: apparently, first item (point-ID) can be a string..
		point_scalars_min = std::min(point_scalars_min,point_scalars[i]);
		point_scalars_max = std::max(point_scalars_max,point_scalars[i]);
		min_p.x = std::min(min_p.x,p.x);
		min_p.y = std::min(min_p.y,p.y);
		max_p.x = std::max(max_p.x,p.x);
		max_p.y = std::max(max_p.y,p.y);
	}
	fclose(fp);

	

	Point2d range(max_p-min_p);

	for(int i=0;i<NP;++i)											//Normalize read points within available image space (fboSize)
	{
		Point2d& p = points[i];
		p.x = (p.x-min_p.x)/range.x*(1-2*t)*fboSize + t*fboSize;
		p.y = (p.y-min_p.y)/range.y*(1-2*t)*fboSize + t*fboSize;
	}

	for(int i=0;i<NP;++i)
	{
		Point2d& p = points[i];
		min_p.x = std::min(min_p.x,p.x);
		min_p.y = std::min(min_p.y,p.y);
		max_p.x = std::max(max_p.x,p.x);
		max_p.y = std::max(max_p.y,p.y);
	}

	//printf("\n$$$ after normalize, X[%f,%f], Y[%f,%f]\n",min_p.x,min_p.y,max_p.x,max_p.y);
	/////// error

	fp = fopen(fer.c_str(),"r");									//2. Read projection-error matrix:
	if (!fp)
	{
		cout<<"Error: Cannot open"<<fer<<endl;
		return false;
	}

	distmatrix = new DistMatrix(NP);
	int nrows;
	fscanf(fp,"%d",&nrows);
	if (nrows!=NP)
	{
		cout<<"Error: error-matrix #rows "<<nrows<<" != #points "<<NP<<endl;
		return false;
	}

	for(int i=0;i<NP;++i)
	  for(int j=0;j<=i;++j)
	  {
		float val;
		if (j<i) fscanf(fp,"%f;",&val);								//Take care, last elem on line not followed by ';'
		else														//Also, note that the matrix stored in the file is symmatric lower-diagonal
		{															//REMARK: We could store this more compactly..
			fscanf(fp,"%f",&val);
			continue;
		}

		(*distmatrix)(i,j) = val;
		(*distmatrix)(j,i) = val;
	  }
	fclose(fp);

	distmatrix->minmax();											//3. Compute range of distance matrix
																	//   -the range [min,0]: false positives (points too close in 2D w.r.t. nD)
																	//   -the range [0,max]: false negatives (points too far in 2D w.r.t. nD)
	cout<<"\nError matrix: ["<<distmatrix->min()<<","<<distmatrix->max()<<"]\n"<<endl;

	if (load_nd)													//4. Read the nD data values:
	{
		fp = fopen(fnd.c_str(),"r");
		fgets(line,1024,fp);											//Skip first line 'DY'
		int NP_n;
		fscanf(fp,"%d",&NP_n);											//Get #points in file
		if (NP_n!=NP)
		{
		   cout<<"Error: "<<NP_n<<" nD points, "<<NP<<" 2D points"<<endl;
		   return false;
		}

		int ND;
		fscanf(fp,"%d",&ND);											//Get nD point dimensions
		attributes.resize(ND);											//Allocate space for attributes
		attributes_min.resize(ND);
		attributes_max.resize(ND);
		for(int i=0;i<ND;++i)
		{
		   attributes[i] = new vector<float>(NP);
		   attributes_min[i] = 1.0e+6;
		   attributes_max[i] = -1.0e+6;
		}

		char pid_nd[128];
		for(int i=0;i<NP;++i)											//Read all n-D points:
		{
			fscanf(fp,"%[^;];",pid_nd);									//REMARK: Apparently, point-ID can be a string..
			for(int j=0;j<ND;++j)										//Read all n-D dimensions for point 'i'
			{
				float dim_j;
				fscanf(fp,"%f;",&dim_j);
				(*attributes[j])[i] = dim_j;
				attributes_min[j] = std::min(attributes_min[j],dim_j);
				attributes_max[j] = std::max(attributes_max[j],dim_j);
			}

			float label;												//Sanity check: label must be the same in 2D and nD
			fscanf(fp,"%f",&label);
			if (label!=point_scalars[i])
			{
			   cout<<"Error: point "<<i<<"("<<pid_nd<<") has label "<<label<<" in nD and label "<<point_scalars[i]<<" in 2D"<<endl;
			   return false;
			}
		}
		fclose(fp);
	}

	return true;
}



void PointCloud::closestEdges(const Point2d& x,int pid,const Edge*& e1,float& d1,const Edge*& e2,float& d2) const
{
	const Point2d& p = points[pid];							//'center' of the edge-fan we search into
	float      alpha = Point2d::edgeAngle(p,x);				//angle of line from p to current x

	const EdgeMatrix::Row& row = (*sorted_edges)(pid);		//all edges in edge-fan

	if (row.size()<2)
	{
		cout<<"ERROR: point "<<pid<<" has only "<<row.size()<<" edges"<<endl;
		e1 = e2 = 0;
		return;
	}

	int i=0;
	for(;i<row.size();++i)									//search where the current angle falls (note that row is sorted on angles..)
	{
	   if (alpha <= row[i].angle) break;
	}

	i = i % row.size();
	int j = i-1;
	if (j<0) j += row.size();								//next edge after edge 'i'

	e2 = &row[i];
	e1 = &row[j];
	d2 = x.distance2line(p,points[e2->pid]);
	d1 = x.distance2line(p,points[e1->pid]);
}



float PointCloud::blendDistance(const Point2d& pix,const Triangle& tr) const
{
	int i0 = tr(0);
	int i1 = tr(1);
	int i2 = tr(2);

	const Point2d& p0 = points[i0];						//closest cloud-point to 'pix'
	const Point2d& p1 = points[i1];
	const Point2d& p2 = points[i2];

	float d1 = pix.distance2line(p0,p1);
	float d2 = pix.distance2line(p0,p2);
	float d3 = pix.distance2line(p1,p2);
	float tmin = min(d1,min(d2,d3));					//DT of triangle

	Point2d c = Point2d::center(p0,p1,p2);
	float dmin = c.dist(pix);							//DT of center
	float B = 0.5*(((dmin)? min(tmin/dmin,1.0f):1.0f) + ((tmin)? max(1-dmin/tmin,0.0f):0.0f));
	return pow(B,0.5f);
}


/*
bool PointCloud::findTriangle(const Point2d& x,int& pid,const Edge*& e1,const Edge*& e2) const
{
	vector<int> nbs;
	searchNN(x,1,nbs);
	pid = nbs[0];															//Closest point to 'x'

	std::deque<int> q;
	std::vector<bool> visited(triangles.size());

	const TrisOfPoint& fan = point2tris[pid];
	for(TrisOfPoint::const_iterator it=fan.begin();it!=fan.end();++it)
	{   }

	while(q.size())
	{
		int t  = *(q.begin()); q.pop_front();
		const Triangle& tr = triangles[t];
		int p0 = tr.idx[0];
		int p1 = tr.idx[1];
		int p2 = tr.idx[2];
		if (x.inTriangle(points[p0],points[p1],points[p2]))
		{
			return hitTriangle(x,pid,e1,e2);
		}

		const TrisOfPoint& f0 = point2tris[p0];
		for(TrisOfPoint::const_iterator it=f0.begin();it!=f0.end();++it)
		   if (!visited[*it]) { q.push_back(*it); visited[*it]=true; }

		const TrisOfPoint& f1 = point2tris[p1];
		for(TrisOfPoint::const_iterator it=f1.begin();it!=f1.end();++it)
		   if (!visited[*it]) { q.push_back(*it); visited[*it]=true; }

		const TrisOfPoint& f2 = point2tris[p2];
		for(TrisOfPoint::const_iterator it=f2.begin();it!=f2.end();++it)
		   if (!visited[*it]) { q.push_back(*it); visited[*it]=true; }

	}

	return false;
}
*/




inline float errorDistWeighting(float d,float delta)
{
   return (d<delta)? 1 : exp(-0.3*(d-delta)*(d-delta));
}


float PointCloud::interpolateDistMatrix(const Point2d& pix,float& certainty,float delta) const
{
	int tid = hitTriangle(pix);
	if (tid<0)										//pix outside the point cloud triangulation: nothing to show there, really
	{
		certainty = 0;
		return 0;
	}

	const Point2d* p[3]; float d[3]; int i[3]; float w[3];
	const DistMatrix& dm = *distmatrix;
	const Triangle& tr = triangles[tid];

	i[0] = tr(0);
	i[1] = tr(1);
	i[2] = tr(2);

	p[0] = &points[i[0]];							//p1 = other end of edge (p,p1)
	p[1] = &points[i[1]];							//p2 = other end of edge (p,p2)
	p[2] = &points[i[2]];
													//Get edge weights:
	w[0] = -std::min(dm(i[2],i[0]),0.0f);			//As we ONLY want to emphasize false-positives here,
	w[1] = -std::min(dm(i[2],i[1]),0.0f);			//so we simply eliminate false-negatives from this data.
	w[2] = -std::min(dm(i[0],i[1]),0.0f);

	d[0] = pix.distance2line(*p[2],*p[0]);
	d[1] = pix.distance2line(*p[2],*p[1]);
	d[2] = pix.distance2line(*p[0],*p[1]);

	w[0] *= errorDistWeighting(p[2]->dist(*p[0]),delta);
	w[1] *= errorDistWeighting(p[2]->dist(*p[1]),delta);
	w[2] *= errorDistWeighting(p[0]->dist(*p[1]),delta);

	certainty = 1;
	if (d[0]<1) return w[0];						//point on (p,p1): return w1
	if (d[1]<1) return w[1];						//point on (p,p2): return w2
	if (d[2]<1) return w[2];						//point on (p,p2): return w2

	float B   = blendDistance(pix,tr);
	certainty = 1-B;								//Record the certainty (0 for deepest point in triangle, 1 for being on the edges)

	float l[3],A[3];								//Use barycentric coordinates to interpolate
	l[0] = p[0]->dist(*p[2]);
	l[1] = p[1]->dist(*p[2]);
	l[2] = p[0]->dist(*p[1]);
	A[0] = 1/(d[0]*l[0]);
	A[1] = 1/(d[1]*l[1]);
	A[2] = 1/(d[2]*l[2]);

	float val = (w[0]*A[0]+w[1]*A[1]+w[2]*A[2])/(A[0]+A[1]+A[2]);
	return val;
}



void PointCloud::computeAggregateError(float norm)								//Compute aggregated projection error for a point wrt all its neighbors
{
    if (distmatrix == NULL)
        return;
    
    float aggregate_error_max = 0;
    const int NP = points.size();

    for(int i=0;i<NP;++i)
    {
        const DistMatrix::Row& row = (*distmatrix)[i];
        aggregate_error[i] = 0;
        for(DistMatrix::Row::const_iterator it=row.begin();it!=row.end();++it)
        {
            float val = fabs(*it);
            aggregate_error[i] += val;											
        }
        aggregate_error_max = std::max(aggregate_error_max,aggregate_error[i]);
    }

    //Normalize aggregated error
    for(int i=0;i<NP;++i)														
        aggregate_error[i] /= NP;
    aggregate_error_max /= NP;	

    if (aggregate_error_max<1.0e-6) 
        aggregate_error_max=1;

    cout<<"Aggregate err max: "<<aggregate_error_max<<endl;

    //Normalization to user-specified range
    if (norm)																	
    {
        aggregate_error_max = norm;
    }

    //Normalize aggregated error
    for(int i=0;i<NP;++i)
        aggregate_error[i] = std::min(aggregate_error[i]/aggregate_error_max,1.0f);
}



float PointCloud::averageNeighborDist(int pid) const							//Return average dist to geometric nbs of point 'pid'
{
		const Point2d& pi = points[pid];
		const EdgeMatrix::Row& erow = (*sorted_edges)(pid);
		float avgdist = 0;

		for(EdgeMatrix::Row::const_iterator it = erow.begin();it!=erow.end();++it)
			avgdist += pi.dist(points[it->pid]);
		avgdist /= erow.size();
		return avgdist;
}



void PointCloud::computeLabelMixing()											//Compute per-point label-mix degree (in [0,1]). This tells from how many of the
{																				//point-neighbor-labels (in its triangle-fan) do the point's label differ.
	for(int i=0;i<points.size();++i)
	{
		const Point2d& pi = points[i];											//Compute label-mixing at point 'pi':
		int lbl_i = point_scalars[i];

		vector<int> nn;															//Get R-nearest neighbors, and see how much mixing we've got in there
		const float rad = avgdist;
		int NN = searchR(pi,rad,200,nn);

		float wsum = 0;															//Compute weights for all neighbors
		float diff = 0;
		for(int j=0;j<NN;++j)
		{
		  float r  = pi.dist(points[nn[j]]);
		  float w  = exp(-r*r/rad/rad);
		  wsum    += w;

		  int lbl_j = point_scalars[nn[j]];
		  if (lbl_j==lbl_i) continue;
		  diff += w;
		}

		label_mix[i] = diff/wsum;
	}
}



void PointCloud::computeFalseNegatives(const Grouping::PointGroup& grp,float range)
{																				//Compute false-negatives w.r.t. all points in 'grp'
	const int NP = points.size();
	vector<float> rerr(NP,1.0e+6);

	for(Grouping::PointGroup::const_iterator it=grp.begin();it!=grp.end();++it)
	{
		int pid = *it;
		computeFalseNegatives(pid,false);										//Compute FN-error of all points wrt 'pid'
		for(int i=0;i<NP;++i)
		   rerr[i] = std::min(rerr[i],false_negative_error[i]);
	}

	if (!range)																	//Auto-normalization
	{
	   for(int i=0;i<NP;++i) range = std::max(range,rerr[i]);
	   cout<<"False negatives max: "<<range<<endl;
	   if (range<1.0e-6) range = 1;
	}

	for(int i=0;i<NP;++i) false_negative_error[i] = std::min(rerr[i]/range,1.0f);
}


void PointCloud::computeFalseNegatives(int pid,bool norm)						//Compute false-negatives w.r.t. point 'pid'
{
	float err_max = 0;
	const DistMatrix::Row& row = (*distmatrix)(pid);							//'row' encodes errors of all points w.r.t. 'pid'

	int idx = 0;
	for(DistMatrix::Row::const_iterator it=row.begin();it!=row.end();++it,++idx)
	{
		float val = std::max(*it,0.0f);											//We only want to emphasize false negatives here,
		false_negative_error[idx] = val;										//so we skip false positive points.
		err_max = std::max(err_max,val);
	}

	float err_sum = 0;															//Assign relative error for 'pid' (wrt itself..)
	const EdgeMatrix::Row& erow = (*sorted_edges)(pid);							//as being the average error wrt its neighbors
	for(EdgeMatrix::Row::const_iterator it = erow.begin();it!=erow.end();++it)
	{
		int     j = it->pid;
		err_sum += false_negative_error[j];
	}

	false_negative_error[pid] = err_sum/erow.size();

	if (!norm) return;

	err_max = std::max(err_max,false_negative_error[pid]);						//Normalize FN-error (if so desired)
	if (err_max<1.0e-6) err_max=1;
	for(int i=0;i<points.size();++i)											//Normalize relative error (so it's in [0,1])
	   false_negative_error[i] /= err_max;
}



void PointCloud::myFitToSize(float minX, float minY, float maxX, float maxY) {
    cout<<"\n"<<"--------"<<__PRETTY_FUNCTION__<<endl;
    //Size of the space between points and the window's border (needed for safe DT computations)
    //const float t = 0.04;
    const float t = 0.08;    
    int NP = points.size();

    //Read all 2D point projections:
    float pointsMinX = 1.0e+8;
    float pointsMinY = 1.0e+8;
    float pointsMaxX = -1.0e+8;
    float pointsMaxY = -1.0e+8;
    for(int i=0;i<NP;++i) {
        Point2d& p = points[i];
        pointsMinX = std::min(pointsMinX,p.x);
        pointsMinY = std::min(pointsMinY,p.y);
        pointsMaxX = std::max(pointsMaxX,p.x);
        pointsMaxY = std::max(pointsMaxY,p.y);	
    }
    
    max_p = Point2d(std::max(maxX, pointsMaxX), std::max(maxY, pointsMaxY));
    min_p = Point2d(std::min(minX, pointsMinX), std::min(minY, pointsMinY));
    Point2d range(max_p-min_p);

    //Normalize read points within available image space (fboSize)
    for(int i=0;i<NP;++i) {
        Point2d& p = points[i];
        p.x = (p.x-min_p.x)/range.x *(1-2*t)*fboSize + t*fboSize;
        p.y = (p.y-min_p.y)/range.y *(1-2*t)*fboSize + t*fboSize;
    }

    min_p = Point2d(1.0e+8,1.0e+8);
    max_p = Point2d(-1.0e+8,-1.0e+8);
    
    for(int i=0;i<NP;++i) {
        Point2d& p = points[i];
        min_p.x = std::min(min_p.x,p.x);
        min_p.y = std::min(min_p.y,p.y);
        max_p.x = std::max(max_p.x,p.x);
        max_p.y = std::max(max_p.y,p.y);	
    }
}

string trim(string str) {
    
    str.erase(str.find_last_not_of(" \n\r\t")+1);
    if (!str.empty()) 
        str = str.substr(str.find_first_not_of(" \n\r\t"));
    
    return str;
}

bool fileExists(string filename) {
    ifstream ifile(filename.c_str());
    return (bool)ifile;    
}

vector<string> explode(const string & str, char delim)
{
    vector<string> elems;
    stringstream ss(str);
    string item;
    
    while (std::getline(ss, item, delim)) {
        item = trim(item);
        if (item.length() > 0)
            elems.push_back(item);
    }
    
    return elems;
}

bool PointCloud::myLoadPex(const char* file, const char* proj, bool load_nd)
{
    cout<<"\n"<<"------------"<<__PRETTY_FUNCTION__<<endl;
    //Size of the space between points and the window's border (needed for safe DT computations)
    //const float t = 0.04;
    const float t = 0.08;

    //n-dimensional points file
    string fnd = file; fnd += ".data";  //////segementation.nd
    //2-dimensional projection file
    string f2d = file;
    if (proj) { f2d += "."; f2d += proj; }			
    f2d += ".2d";
	
    //error matrix (wrt projection of n-D points to 2-D points)
    string fer = file;
    if (proj) { fer += "."; fer += proj; }
    fer += ".err";
    char line[5000];
	
    //1. Read 2D projections:
    cout << "\n------------myLoadPex, Reading 2D points file: " + f2d << endl;
    FILE* fp = fopen(f2d.c_str(),"r");
    if (!fp)
    {
        cout<<"------------Error: Cannot open "<<f2d<<endl;
        return false;
    }	
	
    //Skip first line 'DY'
    fgets(line,1024,fp);
    
    //Get #points in file
    int NP;    
    fgets(line, 1024, fp);
    NP = atoi(trim(line).c_str());
    
    //Get point dimensions (should be 2)
    int dim;    
    fgets(line, 1024, fp);
    dim = atoi(trim(line).c_str());
    if (dim!=2)
    {
        cout<<"------------Warning: 2D projection dimension="<<dim<<", expected 2"<<endl;
    }

    //Get the attributes names
    fgets(line, 5000, fp);

    points.resize(NP);
    point_scalars.resize(NP);
    point_names.clear();
    point_scalars_min = min_p.x = min_p.y = 1.0e+8;
    point_scalars_max = max_p.x = max_p.y = -1.0e+8;
	
    //Read all 2D point projections:
    for(int i=0;i<NP;++i)
    {
        Point2d& p = points[i];
        char pointId[100];
        //REMARK: apparently, first item (point-ID) can be a string..
        //fscanf(fp,"%*[^;];%f;%f;%f",&p.x,&p.y,&point_scalars[i]);
        fscanf(fp,"%[^;];%f;%f;%f",&pointId, &p.x, &p.y, &point_scalars[i]);
        strcpy(pointId, trim(pointId).c_str());
        
        //if (!ProjUtil::isNumber(pointId))
            point_names.push_back(pointId);

        point_scalars_min = std::min(point_scalars_min,point_scalars[i]);
        point_scalars_max = std::max(point_scalars_max,point_scalars[i]);
        min_p.x = std::min(min_p.x,p.x);
        min_p.y = std::min(min_p.y,p.y);
        max_p.x = std::max(max_p.x,p.x);
        max_p.y = std::max(max_p.y,p.y);	
    }
    fclose(fp);
		
//    Point2d range(max_p-min_p);
//
//    //Normalize read points within available image space (fboSize)
//    for(int i=0;i<NP;++i)											
//    {
//        Point2d& p = points[i];
//        p.x = (p.x-min_p.x)/range.x*(1-2*t)*fboSize + t*fboSize;
//        p.y = (p.y-min_p.y)/range.y*(1-2*t)*fboSize + t*fboSize;
//    }
	
    for(int i=0;i<NP;++i)
    {
        Point2d& p = points[i];
        min_p.x = std::min(min_p.x,p.x);
        min_p.y = std::min(min_p.y,p.y);
        max_p.x = std::max(max_p.x,p.x);
        max_p.y = std::max(max_p.y,p.y);	
    }
	
    //2. Read projection-error matrix:
    cout << "\n------------myLoadPex, Reading projection-errors file: " + f2d << endl;
    if (!fileExists(fer)) {
        cout << "\n------------Warning: Errors file " << fer << " doesn't exist" << endl;
        distmatrix = NULL;                                     //Create a distance matrix and fill it with dummy data      
    } else {
        fp = fopen(fer.c_str(),"r");									
        if (!fp)
        {
            cout<<"------------Error: Cannot open"<<fer<<endl;
            return false;
        }		

        distmatrix = new DistMatrix(NP);

        int nrows;
        fscanf(fp,"%d",&nrows);
        if (nrows!=NP)
        {
            cout<<"------------Error: error-matrix #rows "<<nrows<<" != #points "<<NP<<endl;
            return false;
        }

        for(int i=0;i<NP;++i)
            for(int j=0;j<=i;++j)
            {
                float val;
                //Take care, last element on line not followed by ';'
                if (j<i)                
                    fscanf(fp,"%f;",&val);
                //Also, note that the matrix stored in the file is symmetric lower-diagonal
                //REMARK: We could store this more compactly..
                else
                {
                    fscanf(fp,"%f",&val);
                    continue;
                }

                (*distmatrix)[i][j] = val;
                (*distmatrix)[j][i] = val;
            }
        fclose(fp);			

        //Compute range of distance matrix
        distmatrix->minmax();
        //   -the range [min,0]: false positives (points too close in 2D w.r.t. nD)
        //   -the range [0,max]: false negatives (points too far in 2D w.r.t. nD)
        cout<<"------------Error matrix: ["<<distmatrix->min()<<","<<distmatrix->max()<<"]"<<endl;
    }
    
    //3 Read the nD data values:
    if (true)
    {
        cout << "\n------------myLoadPex, Reading nD points file: " + fnd << endl;
        fp = fopen(fnd.c_str(),"r");
        //Skip first line 'DY'
        fgets(line,1024,fp);

        //Get #points in file
        int NP_n;
        fgets(line, 1024, fp);
        NP_n = atoi(line);

        if (NP_n!=NP)
        {
           cout<<"------------Error: "<<NP_n<<" nD points, "<<NP<<" 2D points"<<endl;
           return false;
        }
       
        //Get the #attributes
        fgets(line, 512, fp);
        int ND;
        ND = atoi(line);
        
        //Get the attributes names
        fgets(line,5000,fp);
        attributes_names = explode(line, ';');
        attributes_original_names = explode(line, ';');
        
        //Allocate space for attributes
        attributes.resize(ND);
        attributes_min.resize(ND);
        attributes_max.resize(ND);
        attributes_mean.resize(ND, 0.0f);
        
        attributes_ids.resize(ND);
        
        attributes_original.resize(ND);
        attributes_original_min.resize(ND);
        attributes_original_max.resize(ND);
        attributes_original_mean.resize(ND, 0.0f);
        
        for(int i=0;i<ND;++i) {
            attributes[i] = new vector<float>(NP);
            attributes_min[i] = 1.0e+6;
            attributes_max[i] = -1.0e+6;
            
            attributes_ids[i] = i;
            attributes_indices[i] = i;
            
            attributes_original[i] = new vector<float>(NP);
            attributes_original_min[i] = 1.0e+6;
            attributes_original_max[i] = -1.0e+6;
        }

        char pid_nd[128];
        //Read all n-D points:
        for(int i=0;i<NP;++i)
        {
            //REMARK: Apparently, point-ID can be a string..    
            fscanf(fp,"%[^;];",pid_nd);
            //Read all n-D dimensions for point 'i'
            for(int j=0;j<ND;++j) {
                float dim_j;
                fscanf(fp,"%f;",&dim_j);
                (*attributes[j])[i] = dim_j;
                attributes_min[j] = std::min(attributes_min[j],dim_j);
                attributes_max[j] = std::max(attributes_max[j],dim_j);
                attributes_mean[j] += dim_j;
                
                (*attributes_original[j])[i] = dim_j;
                attributes_original_min[j] = std::min(attributes_min[j],dim_j);
                attributes_original_max[j] = std::max(attributes_max[j],dim_j);
                attributes_original_mean[j] += dim_j;                
            }
            
            //Sanity check: label must be the same in 2D and nD
            float label;
            fscanf(fp,"%f",&label);
            if (label!=point_scalars[i])
            {
                cout<<"            Error: point "<<i<<"("<<pid_nd<<") has label "<<label<<" in nD and label "<<point_scalars[i]<<" in 2D"<<endl;
                return false;
            }
        }
        fclose(fp);
        
        for (int j = 0; j < ND; j++) {
            attributes_mean[j] = attributes_mean[j] / ND;
            attributes_original_mean[j] = attributes_original_mean[j] / ND;
        }
        for (int j = 0; j < 5; j++) {
            printf("\n------------attributes_min=%f\n",attributes_min[j]);
        }
        cout << "\n------------myLoadPex, Computing squared distance matrix... " << endl;
        //Compute the squared distance matrix in nd
        sqrDistanceMatrix = new DistMatrix(NP);        
        for (int i = 0; i < NP; i++) {                 
            for (int j = i; j < NP; j++) {
                double dij = 0.0;    
                for (int dim = 0; dim < ND; dim++) {
                    float pointDimValue = (*attributes[dim])[i];
                    float pointNormalized = 0.0f;
                    if (attributes_max[dim] - attributes_min[dim] > 0.0001)            
                        pointNormalized = (pointDimValue - attributes_min[dim]) / (attributes_max[dim] - attributes_min[dim]);

                    float neighborDimValue = (*attributes[dim])[j];
                    float neighborNormalized = 0.0f;
                    if (attributes_max[dim] - attributes_min[dim] > 0.0001)
                        neighborNormalized = (neighborDimValue - attributes_min[dim]) / (attributes_max[dim] - attributes_min[dim]);            

                    dij = dij + pow(pointNormalized - neighborNormalized, 2);
                }
                (*sqrDistanceMatrix)[i][j] = dij;
                (*sqrDistanceMatrix)[j][i] = dij;                
            }
        }
        //cout << "\n------------done!" << endl;
    }
	
    return true;
}

 

void PointCloud::dimensionRank(float radius) {    
    cout<<"\n"<<"------------"<<__PRETTY_FUNCTION__<<endl;
    radius = radius * diameter;
      
    //Stores the ranking of all individual points
    point_dimrank.clear();
    point_dimrank.resize(points.size());

    //Stores the overall importance of each dimension to this cloud
    dimHistogram.clear();
    for (int i = 0; i < attributes.size(); i++)
        dimHistogram.push_back(DimensionHistogram(attributes_ids[i], 0));
    
    printf("\n----------------Loop dimensionRankAvg()\n");
    printf("\n--------------------Loop dimRankContributionAvg()\n");
    for (int i = 0; i < points.size(); i++) {
        
        DimRankingList pointRanks = dimensionRankAvg(i, radius);
        //dimensionRank
        //dimensionRankContribution
        point_dimrank[i] = pointRanks;
        if (!pointRanks.empty()) {
            int dimIndex = attributes_indices[pointRanks[0].dimId];
            dimHistogram[dimIndex].frequency += 1;
        }
    }
    std::sort(dimHistogram.begin(), dimHistogram.end(), DimensionHistogram::SortByFrequencyDesc());
    cout << "\n------------dimensionRank finished..." << endl; 
    return;
}

DimRankingList PointCloud::dimensionRankAvg(int pid, float radius) {
    //循环内,N个输出
    //printf("\n------------DimRankingList PointCloud::dimensionRank(int pid, float radius)\n");
    //1 - Get the n neighbors of pid
    vector<int> closestNeighbors;
    //Search for the closest k nearest neighbors of pid
    //this->searchNN(points[pid], neighborsCount, closestNeighbors, &neighborsDistance);    
    //Search pid neighbors based in a radius
    this->searchR(points[pid], radius, points.size(), closestNeighbors);
    //Remove the 1st element (is own pid)
    closestNeighbors.erase(closestNeighbors.begin());
    
    //2 - Find the attributes importance weights for each neighbor
    DimRankingList pidRanks;
    if (closestNeighbors.size() < dimrank_min_group_size)
        return pidRanks;
    
    switch (dimrank_metric) {
        case DIMRANK_CONTRIBUTION:
            pidRanks = dimRankContributionAvg(pid, closestNeighbors, DIMRANK_SIMILARITY);
            break;
        //case DIMRANK_VARIANCE:
        //    pidRanks = dimensionRankVariation(pid, closestNeighbors, DIMRANK_SIMILARITY);
        //    break;
        //case DIMRANK_STORED:
        //    pidRanks = dimensionRankTSNEStored(pid);
//        case DIMRANK_PCA:
//            pidRanks = dimensionRankPCA(pid, closestNeighbors);
//            break;
        ///case DIMRANK_TEMP:
        ///    pidRanks = dimensionRankContribution2(pid, closestNeighbors);
        ///    break;
    };
               
    return pidRanks;
}

DimRankingList PointCloud::dimRankContributionAvg(int pid, std::vector<int> &pointList, int strategy) {
    //循环内,N个输出
    //printf("\n------------DimRankingList PointCloud::dimensionRankContribution(int pid, std::vector<int> &pointList, int strategy)\n");
    //1 - Initialization
    int ND = attributes.size();
    DimRankingList pidRanks(ND);
    for (int dim = 0; dim < ND; dim++)
        pidRanks[dim].dimId = dim;
    
    float allDimsSum = 0.0f;                                                    //normalization-purpose variable
    //2 - Compute the average contribution for similarity/dissimilarity of this group
    //    to each dimension [0..ND-1]
    for (int dim = 0; dim < ND; dim++) {
        
        //3 - Normalize current dimension first
        float dimMaxMinDist = attributes_max[dim] - attributes_min[dim];
        vector<double> normalizedNeighbors(pointList.size(), 0.0);
        double normalizedPid = 0.0;
        
        //3.1 - Normalize only if there is a significative distance between max and min value for this dimension
        if (dimMaxMinDist > 0.00001) {
            for (int neighbor = 0; neighbor < pointList.size(); neighbor++) {
                int neighborId = pointList[neighbor];
                float neighborDimValue = (*attributes[dim])[neighborId];
                float neighborNormalized = (neighborDimValue - attributes_min[dim]) / dimMaxMinDist;            

                normalizedNeighbors[neighbor] = neighborNormalized;
            }           
            normalizedPid =  ((*attributes[dim])[pid] - attributes_min[dim]) / dimMaxMinDist;
        }

        //4 - Stores, for each neighbor, the contribution of this dimension to the final distance
        vector<double> neighborsRelativeDistances(pointList.size(), 0.0);
        double neighborsDimSum = 0.0;
        for (int neighbor = 0; neighbor < pointList.size(); neighbor++) {
            int neighborId = pointList[neighbor];
            float ndDistance = (*sqrDistanceMatrix)[pid][neighborId];
            float dimDistance = std::pow(normalizedPid - normalizedNeighbors[neighbor], 2);
            float relativeDimDistance = dimDistance / ndDistance;
            
            neighborsRelativeDistances[neighbor] = relativeDimDistance;
            neighborsDimSum += neighborsRelativeDistances[neighbor];
        }
        //5 - Compute the average relative distance contribution  
        pidRanks[dim].weight = neighborsDimSum / pointList.size();
        if (centroidRanks[dim].weight > 0.00000)
            pidRanks[dim].weight = pidRanks[dim].weight / centroidRanks[dim].weight;
        else
            pidRanks[dim].weight = 0;

        //Compute the total weight of all dimensions (just to check if it will sum to 1 in the end of this dim loop)
        //or to normalization purposes
        allDimsSum += pidRanks[dim].weight;
    }

    //Normalize, so it will sum up to 1
    float normalizedWeight = 0.0;
    for (int i = 0; i < ND; i++) {
        pidRanks[i].weight = pidRanks[i].weight / allDimsSum;
        normalizedWeight += pidRanks[i].weight;
    }
    
    
    //Sorting in ascending order, so the first is the most important dimension to describe similarity
    if (strategy == DIMRANK_SIMILARITY)
        std::sort(pidRanks.begin(), pidRanks.end(), DimensionRank::SortByWeightAsc());
    else
        std::sort(pidRanks.begin(), pidRanks.end(), DimensionRank::SortByWeightDesc());
        
    return pidRanks;
}



void PointCloud::dimensionRankCentroid() {
    cout<<"\n"<<"------------"<<__PRETTY_FUNCTION__<<endl;
  
    int ND = attributes.size();
    centroidRanks.clear();
    centroidRanks.resize(ND);
    for (int dim = 0; dim < ND; dim++)
        centroidRanks[dim].dimId = dim;
    
    vector<float> sqrDistance(points.size(), 0.0f);
    for (int i = 0; i < points.size(); i++) {
        double dij = 0.0;    
        for (int dim = 0; dim < ND; dim++) {
            float pointDimValue = attributes_mean[dim];
            float pointNormalized = 0.0f;
            if (attributes_max[dim] - attributes_min[dim] > 0.0001)            
                pointNormalized = (pointDimValue - attributes_min[dim]) / (attributes_max[dim] - attributes_min[dim]);

            float neighborDimValue = (*attributes[dim])[i];
            float neighborNormalized = 0.0f;
            if (attributes_max[dim] - attributes_min[dim] > 0.0001)
                neighborNormalized = (neighborDimValue - attributes_min[dim]) / (attributes_max[dim] - attributes_min[dim]);            

            dij = dij + pow(pointNormalized - neighborNormalized, 2);
        }
        sqrDistance[i] = dij;
    }

    
    float allDimsSum = 0.0f;
    for (int dim = 0; dim < ND; dim++) {
        float dimMaxMinDist = attributes_max[dim] - attributes_min[dim];
        vector<double> normalizedNeighbors(points.size(), 0.0);
        double normalizedPid = 0.0;
        
        //If there is a significative distance between max and min value for this dimension, normalize it
        if (dimMaxMinDist > 0.00001) {
            for (int neighborId = 0; neighborId < points.size(); neighborId++) {
                float neighborDimValue = (*attributes[dim])[neighborId];
                float neighborNormalized = (neighborDimValue - attributes_min[dim]) / 
                                         (attributes_max[dim] - attributes_min[dim]);            

                normalizedNeighbors[neighborId] = neighborNormalized;
            }
            normalizedPid =  attributes_mean[dim] - attributes_min[dim];
            normalizedPid /= attributes_max[dim] - attributes_min[dim];
        }

        //Stores, for each neighbor, the contribution of this dimension to the final distance
        vector<double> neighborsRelativeDistances(points.size(), 0.0);
        double neighborsDimSum = 0.0;
        for (int neighborId = 0; neighborId < points.size(); neighborId++) {
            float ndDistance = sqrDistance[neighborId];
            float dimDistance = std::pow(normalizedPid - normalizedNeighbors[neighborId], 2);
            float relativeDimDistance = dimDistance / ndDistance;
            
            neighborsRelativeDistances[neighborId] = relativeDimDistance;
            neighborsDimSum += neighborsRelativeDistances[neighborId];
        }
        //Compute the average relative distance contribution from all neighbors  
        centroidRanks[dim].weight = neighborsDimSum / points.size();
        //Compute the total weight fro all dimensions (just to check if it will sum to 1 in the end of this dim loop)
        allDimsSum += centroidRanks[dim].weight;
    }

    //Sorting in ascending order, so the first is the most important dimension to describe similarity
    //std::sort(centroidRanks.begin(), centroidRanks.end(), DimensionRank::SortByWeightAsc());    
    cout << "\n------------Centroid ranking complete" << endl;
}



void PointCloud::myReduceAttributes(int toDimensionality) {
    
    //1 - Check if variances were computed. If not, compute it
    if (attributes_original_variance.empty()) {
        attributes_original_variance.resize(attributes_original.size());
        for (int i = 0; i < attributes_original.size(); ++i) {
            vector<double> v((*attributes_original[i]).begin(),(*attributes_original[i]).end());
            //Normalizing
            for (int j = 0; j < v.size(); j++)
                v[j] = (v[j] - attributes_original_min[i])/(attributes_original_max[i] - attributes_original_min[i]);
            attributes_original_variance[i] = variance(v);    
        }
    }
    
    //2 - Sort variances by desc order
    vector<pair<int, float> > dimVariancePair;
    for (int i = 0; i < attributes_original_variance.size(); i++) {
        dimVariancePair.push_back(std::make_pair(i, attributes_original_variance[i]));
    }
    std::sort(dimVariancePair.begin(), dimVariancePair.end(), sort_pair_second_desc<int, float>());
       
    int ND = (int)attributes_original.size();
    int reducedND = std::min((int)attributes_original.size(), toDimensionality);
    attributes.clear();
    attributes_min.clear();
    attributes_max.clear();
    attributes_mean.clear();

    int NP = attributes_original[0]->size();
    
    attributes_names.clear();
    attributes_ids.clear();
    attributes_indices.clear();
    
    for(int i=0; i < ND; i++) {
        
        bool dimAmongMostVariance = false;
        for (int k = 0; k < reducedND; k++)
            if (dimVariancePair[k].first == i)
                dimAmongMostVariance = true;

        if (!dimAmongMostVariance)
            continue;
            
        attributes.push_back(attributes_original[i]);
        attributes_min.push_back(attributes_original_min[i]);
        attributes_max.push_back(attributes_original_max[i]);
        attributes_mean.push_back(attributes_original_mean[i]);
        attributes_names.push_back(attributes_original_names[i]);        
        
        attributes_indices[i] = attributes_ids.size();
        attributes_ids.push_back(i); 
    }    
    
    attributes_variance.resize(reducedND);
    for (int i = 0; i < attributes.size(); ++i) {
        vector<double> v((*attributes[i]).begin(),(*attributes[i]).end());
        //Normalizing
        for (int j = 0; j < v.size(); j++)
            v[j] = (v[j] - attributes_min[i])/(attributes_max[i] - attributes_min[i]);
        attributes_variance[i] = variance(v);    
    }    
}
 


void PointCloud::computeTopDims() {

    cout<<"\n"<<"--------------------"<<__PRETTY_FUNCTION__<<endl;
 
    int currentCmapBak = current_cmap;
    int invert_colormapBak = invert_colormap;
    invert_colormap = 0;
	colorMap.load(CMAP_CATEGORICAL);
    //printf("--------1--------");

    dimrank_topdims.clear();
    int NP = points.size();
    int numColors;
    if (attributes.size() > colorMap.getSize() - 1)
        numColors = colorMap.getSize() - 1;
    else
        numColors = attributes.size(); ////attributes.size()=0!!!!!

    //2 - Find the top-ranked (# of colors) dimensions, using the histogram
    //sorted in desc frequency order.
    //printf("\nattributes.size()=%d,colorMap.getSize()=%d\n",attributes.size(),colorMap.getSize()); /////// = 0 错误!!
    for (int i = 0; i < numColors; i++) {
        printf("\n--------------------dimHistogram[%d].frequency=%d\n",i,dimHistogram[i].frequency);
        if (dimHistogram[i].frequency > 0) {
            int dimId = dimHistogram[i].dimId;
            dimrank_topdims.push_back(dimId);
            
        }
    }
    //printf("--------2--------");
    invert_colormap = invert_colormapBak;
    colorMap.load(currentCmapBak);
    //printf("--------3--------");
}

void PointCloud::filterRankings(bool topdims_only) {

cout<<"\n"<<"----------------"<<__PRETTY_FUNCTION__<<endl;
float dimrank_filter_size = 0.7f;
float dimrank_radius = 0.1f;
    float filter_radius = dimrank_filter_size * dimrank_radius * this->diameter;
    if (dimrank_filter_size <= 0.0f)
        return;
    
    if (topdims_only && dimrank_topdims.empty()){
        printf("\n----------------dimrank_topdims = empty\n");
        computeTopDims();
    }
        
    
    this->point_dimrank_visual.clear();
    float minContribution = 1.0f;
    float maxContribution = 0.0f;
    int ND = this->attributes.size();    
    int NP = this->points.size();
    //printf("==========1========");
    //Main idea: Visit every point, get its neighbors (based on the filter_radius) 
    //and check what are their winner dimensions. Store its weights on a vector of pairs
    for (int i = 0; i < NP; i++) {

        DimensionRank currentVisualRank;
        currentVisualRank.dimId = -1;
        //currentVisualRank.weight = -1;
        currentVisualRank.weight = 0;

        //1 - If ranked as invalid, don't change anything on its ranking
        if (this->point_dimrank[i].empty()) {                        
            this->point_dimrank_visual.push_back(currentVisualRank);                
            continue;
        }
        
        //2 - If using only most important dims and this topRankedDim is not among them, proceed to the next point
        int topRankedDim = this->point_dimrank[i][0].dimId;
        if (topdims_only && (std::find(dimrank_topdims.begin(), dimrank_topdims.end(), topRankedDim) == dimrank_topdims.end())) {          
            this->point_dimrank_visual.push_back(currentVisualRank);   
            continue;
        }
        //printf("=========2=========");
        //3 - Get its neighbors (based on the filter_radius)
        const Point2d& pi = this->points[i];       
        vector<int> closestNeighbors;
        this->searchR(pi, filter_radius, NP-1, closestNeighbors);

        //3.1 - If no neighbors were found, don't change anything on this point rankings
        //      and proceed to the next point
        if (closestNeighbors.size() == 1) {                
            currentVisualRank.dimId = this->point_dimrank[i][0].dimId;
            currentVisualRank.weight = this->point_dimrank[i][0].weight;
            this->point_dimrank_visual.push_back(currentVisualRank);  
            continue;
        }

        //4 - Create a vector of pairs to store the top-ranked dimensions {ids, weights} for this neighborhood
        std::vector<std::pair<int,float> > neighborsTopDimFilter;
        for (int topDimIndex = 0; topDimIndex < ND; topDimIndex++) {
            neighborsTopDimFilter.push_back(std::make_pair(attributes_ids[topDimIndex], 0.0f));
        }                     

        //printf("========3==========");

        //4.1 - For every neighbor, sum up its top-ranked weights
        float totalSum = 0.0;
        for(int j = 0; j < closestNeighbors.size(); j++) {
                
            int nbIndex = closestNeighbors[j];
            if (this->point_dimrank[nbIndex].empty())
                continue;
                
            int nbTopDimId = this->point_dimrank[nbIndex][0].dimId;     
            //if the top-ranked dimension is not among the most important, ignore this neighbor
            if (topdims_only && (std::find(dimrank_topdims.begin(), dimrank_topdims.end(), nbTopDimId) == dimrank_topdims.end()))            
                continue;
            
            int nbTopDimIndex = attributes_indices[nbTopDimId];
            //Accumulate the weight of the top-ranked dim 
            ////if (dimrank_strategy == DIMRANK_SIMILARITY) {
                neighborsTopDimFilter[nbTopDimIndex].second += (1 - this->point_dimrank[nbIndex][0].weight);
                totalSum = totalSum + (1 - this->point_dimrank[nbIndex][0].weight);
            ////} else {
            ////    neighborsTopDimFilter[nbTopDimIndex].second += this->point_dimrank[nbIndex][0].weight;
            ////    totalSum = totalSum + this->point_dimrank[nbIndex][0].weight;                        
            ////}
                
        }
        //4.2 - Sort the vector by weight in desc order
        std::sort(neighborsTopDimFilter.begin(), neighborsTopDimFilter.end(), //Sort the set frequencies array in desc order
            sort_pair_second<int, float, std::greater<float> >());

        //4.3 - Get the top-ranked dimension, and normalize its weight
        int filteredDimWinner = neighborsTopDimFilter[0].first;
        float filteredDimWeight = 0.0;
        if (totalSum > 0.0f) {
            filteredDimWeight = neighborsTopDimFilter[0].second / totalSum;
        }
        else {
            filteredDimWinner = -1;
        }
            
//        if (filteredDimWinner < colorMap.getSize()-1)
            currentVisualRank.dimId = filteredDimWinner;
//        else
//            currentVisualRank.dimId = -1;
            
        currentVisualRank.weight = filteredDimWeight;
        this->point_dimrank_visual.push_back(currentVisualRank);
                
        if (filteredDimWeight < minContribution)
            minContribution = filteredDimWeight;
        if (filteredDimWeight > maxContribution)
            maxContribution = filteredDimWeight;
    }

    //5 - Assign the min contribution to invalid and 'others' instances
    for (int i = 0; i < this->points.size(); i++) {

        if (this->point_dimrank_visual[i].dimId == -1) {
                ////point_dimrank_visual[i].weight = maxContribution * 0.85f;
            point_dimrank_visual[i].weight = 0.6f; 
        }

        
        if (this->point_dimrank[i].empty()) {
            //this->point_dimrank_visual[i].weight = minContribution;
            this->point_dimrank_visual[i].weight = 0;
        }

    }
}





















