#include "include/config.h"
#include "include/skelft.h"
#include "include/vis.h"
#include "include/myrandom.h"
#include "include/pointcloud.h"
#include "include/sparsematrix.h"
#include "include/fullmatrix.h"
#include "include/grouping.h"
#include "include/io.h"
#include "include/gdrawing.h"

#include <math.h>
#include <iostream>
#include <string>
#include <time.h>
#include "include/cudawrapper.h"
#include <cuda_gl_interop.h>

using namespace std;





//-------------------------------------------------------------------------------------------------------


void initPointCloud(PointCloud*,int,int);
void testPointCloud(PointCloud*,int);
void loadParameters(int argc, char **argv); 
PointCloud* loadPointCloud();
//GraphDrawing* loadDynamicProjectionGraph();
//void fitCloudToGraphDimensions(PointCloud* cloud, GraphDrawing* graph);


//-------------------------------------------------------------------------------------------------------


//RadialGrouping*     visual_clustering = 0;
//Grouping*			labelg = 0;
int   NP	    = 10;
int   fboSize   = 1024;
bool  load_nd   = false;
bool  load_trails = false;
bool  load_rankings = false;
bool  compute_rankings_offline = false;

int main(int argc,char **argv)
{

    //1. Load parameters from command line
    loadParameters(argc, argv);
	printf("\n$$$ after loadParameters. pointfile = %s\n",pointfile);


    //Let CUDA communicate with OpenGL

    colorMap.load(current_cmap);
 	printf("\n$$$ test colorMap.load, get(1) = (%f,%f,%f)\n",colorMap.getColor(1).r,colorMap.getColor(1).g,colorMap.getColor(1).b);



    //Initialize CUDA DT/FT API
    skelft2DInitialization(fboSize);									
	printf("\n$$$ not concerned about skelft2DInitialization()\n");





    //2. Load dynamic projection graph
    ////GraphDrawing* dynamic_projection  = 0;   
    ////if (load_trails) 
    ////    dynamic_projection = loadDynamicProjectionGraph();
        
    //3. Point cloud
    PointCloud*	fullCloud = loadPointCloud();

	printf("\n$$$ after load, X[%f,%f], Y[%f,%f]\n",fullCloud->min_p.x,fullCloud->min_p.y,fullCloud->max_p.x,fullCloud->max_p.y);
    


    ////if (load_trails) {
    ////    fitCloudToGraphDimensions(fullCloud, dynamic_projection);
    ////    dynamic_projection->normalize(Point2d(fboSize,fboSize),0.08);
    ////} 
    ////else
      fullCloud->myFitToSize();
	  printf("\n$$$ after fitsize, X[%f,%f], Y[%f,%f]\n",fullCloud->min_p.x,fullCloud->min_p.y,fullCloud->max_p.x,fullCloud->max_p.y);
    
          
    //Finalize point cloud creation, once all points are added
    fullCloud->initEnd();															
    
    //Remove trivial exact-overlaps of points (since they create stupid visualization problems)
    //RadialGrouping* rg = new RadialGrouping(fullCloud);						
    //From now on, use only the cleaned-up points	
    //PointCloud* cleanCloud = rg->coarsen(0)->cloud;								
    
    //delete fullCloud;
    //delete rg;
    
    
    
//    int tempArray[] = {5,12,122,134,149,128,123,110,302,89,3,154,175,164,171,116};
//    vector<int> temp;
//    for (int i = 0; i < 16; i++)
//        temp.push_back(tempArray[i]);
//    
//    
//    fullCloud->dimensionRank(0.1f);
//    fullCloud->filterRankings(true);
//    
//    PointCloud *subCloud1 = fullCloud->cloudCopy(temp);
//    subCloud1->avgdist = point_influence_radius;
//    subCloud1->initEnd();
//    		
    
    printf("\n$$$ initEnd finished\n");
    //Initialize visualization engine
    Display* dpy = new Display(1,fboSize, fullCloud, argc, argv);
    dpy->selected_point_id = selected_point_id;
    glutMainLoop(); 	

    skelft2DDeinitialization(); 
    //delete vg;
    //delete labelg;
//    delete rg;
    delete dpy;
    
    //delete cloud;
    
    return 0;
}


PointCloud* loadPointCloud() {

    PointCloud *cloud = new PointCloud(fboSize);					

	//////////////appear errors alloc

    //Read data from file:
    if (pointfile)
    {
        cout << "\nReading PEx file (loadPointCloud)..." << endl;
        bool ok = false;
        if (load_trails) { ////////////默认false
            char newPointFileName[100];
            sprintf(newPointFileName, "%s.0", pointfile);
            ok = cloud->myLoadPex(newPointFileName, projname, load_nd);
        }
        else
            ok = cloud->myLoadPex(pointfile, projname, load_nd);
        if (!ok)
        {
            cout<<"Cannot read given pointcloud data, aborting"<<endl;
            exit(1);
        }
        cout << "Finished reading PEx file." << endl;
    }
    //Generate synthetic data:
    else
    {
       if (NP>0)
           //Initialize with random point distribution
           initPointCloud(cloud,fboSize,NP);								
       else
           //Create simple point-cloud with 3 points (for testing)
           testPointCloud(cloud,fboSize);
    }    
    
    return cloud;
}

void loadParameters(int argc, char **argv) {

    for (int ar=1;ar<argc;++ar)
    {
        string opt = argv[ar];
        if (opt=="-n")
        {
            ++ar;
            NP = atoi(argv[ar]);
        }
        else if (opt=="-f")
        {
            ++ar;
            pointfile = argv[ar];
            if (ar+1<argc && argv[ar+1][0]!='-')
            {
                ++ar;
                projname = argv[ar];
            }
        }
        else if (opt=="-i")
        {
            ++ar;
            fboSize = atoi(argv[ar]);
        }
        else if (opt=="-d")
        {
            load_nd = true;
        }
        else if (opt == "-r") {
            load_rankings = true;
        }
        else if (opt == "-p") {
            compute_rankings_offline = true;
        }
        else if (opt == "-t") {
            load_trails = true;
            ++ar;
            timeframes_total = atoi(argv[ar]);
        }
    }    
}





int main1(int argc,char **argv)
{
	/*
	int   NP		= 10;											//#particles to use (default)
	char* pointfile = 0;
	char* projname  = 0;
	int	  fboSize   = 1024; /////////chushi size
	bool  load_nd   = false;

	for (int ar=1;ar<argc;++ar)
	{
		string opt = argv[ar];
		if (opt=="-n")
		{
			++ar;
			NP = atoi(argv[ar]);
		}
		else if (opt=="-f")
		{
			++ar;
			pointfile = argv[ar];
			if (ar+1<argc && argv[ar+1][0]!='-')
			{
			   ++ar;
			   projname = argv[ar];
			}
		}
		else if (opt=="-i")
		{
			++ar;
			fboSize = atoi(argv[ar]);
		}
		else if (opt=="-d")
		{
			load_nd = true;
		}
	}

	glutInitWindowSize(fboSize, fboSize);								//Graphics system initialization: This must occur in a very specific order!
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA);
	glutInit(&argc, argv);															//  1. first initialize GLUT
  int mainWin = glutCreateWindow("Graph bundling");	  //  2. then create one main window, which initializes OpenGL
	glewInit();                                         //  3. then initialize GLEW

  skelft2DInitialization(fboSize);									  //Initialize CUDA DT/FT API



	PointCloud*		cloud = new PointCloud(fboSize);			//2. Point cloud

	if (pointfile)														//Read data from file:
	{
	   bool ok = cloud->loadPex(pointfile,projname,load_nd);
	   if (!ok)
	   {
		  cout<<"Cannot read given data, aborting"<<endl;
		  return 1;
	   }
	}
	else																//Generate synthetic data:
	{
	   if (NP>0)
		  initPointCloud(cloud,fboSize,NP);								//Initialize with random point distribution
	   else
	    testPointCloud(cloud,fboSize);								//Create simple point-cloud with 3 points (for testing)
	}

	cloud->initEnd();													//Finalize point cloud creation, once all points are added

	RadialGrouping* rg = new RadialGrouping(cloud);						//Remove trivial exact-overlaps of points (since they create stupid visualization problems)
	RadialGrouping* crs = rg->coarsen(0);								//From now on, use only the cleaned-up points
	PointCloud* clean_cloud = crs->cloud;								//


	RadialGrouping* vg = new RadialGrouping(clean_cloud);				//Make an engine to coarsen the cloud; We'll use it further for simplified visualizations.
	//StronglyConnectedGrouping* vg = new StronglyConnectedGrouping(clean_cloud);
	//vg->build();
	visual_clustering = vg;

	labelg = clean_cloud->groupByLabel();

  Display* dpy = new Display(mainWin,fboSize,clean_cloud,argc,argv);			//Initialize visualization engine
  glutMainLoop();

  skelft2DDeinitialization();
	delete vg;
	delete labelg;
	delete rg;
	delete dpy;
	delete cloud;
    return 0;*/
}



void testPointCloud(PointCloud* cloud,int size)
{
	const float t = 0.05;

	cloud->points.resize(3);
	cloud->point_scalars.resize(3);
	cloud->point_scalars_min = 1.0e+8;
	cloud->point_scalars_max = -1.0e+8;
	cloud->distmatrix = new PointCloud::DistMatrix(3);

	float wX = size, wY = size;

	cloud->points[0] = Point2d(t*wX,t*wY);
	cloud->point_scalars[0] = 0;
	cloud->points[1] = Point2d((1-t)*wX,t*wY);
	cloud->point_scalars[1] = 0;
	cloud->points[2] = Point2d((1-t)*wX,(1-t)*wY);
	cloud->point_scalars[2] = 0;

	(*cloud->distmatrix)(0,1) = (*cloud->distmatrix)(1,0) = 1;
	(*cloud->distmatrix)(0,2) = (*cloud->distmatrix)(2,0) = 0.5;
	(*cloud->distmatrix)(1,2) = (*cloud->distmatrix)(2,1) = 0;


	for(int i=0;i<3;++i)
	{
			const Point2d& np = cloud->points[i];
			const float& val = cloud->point_scalars[i];
			if (cloud->point_scalars_min>val) cloud->point_scalars_min = val;
			if (cloud->point_scalars_max<val) cloud->point_scalars_max = val;
			cloud->min_p.x = std::min(cloud->min_p.x,np.x);
			cloud->min_p.y = std::min(cloud->min_p.y,np.y);
			cloud->max_p.x = std::max(cloud->max_p.x,np.x);
			cloud->max_p.y = std::max(cloud->max_p.y,np.y);
	}
}






void initPointCloud(PointCloud* cloud,int size,int NP)					//Some test-initialization of a point cloud
{
	const int   NNBS = 1;												//Number of neighborhoods/clusters to make
	const float NMAX = ceil(float(NP)/NNBS);							//Max # points in a 'cluster'
	const float t = 0.05;

	randinit(clock());													//Initialize random generator to hopefully something random itself..

	cloud->points.resize(NP);
	cloud->point_scalars.resize(NP);
	cloud->point_scalars_min = 1.0e+8;
	cloud->point_scalars_max = -1.0e+8;
	cloud->distmatrix = new PointCloud::DistMatrix(NP);


	float wX = size, wY = size;
	float dX = t*wX, dY = t*wY;
	wX -= 2*dX; wY -= 2*dY;

	float diag  = sqrt(wX*wX+wY*wY);
	float R_max = sqrt(wX*wY/M_PI);

	bool ready = false;
	int  ngen  = 0;
	for(int i=0;!ready && i<NP;++i)
	{
		int ii = dX + myrandom()*wX;										//Center of current neighborhood
		int jj = dY + myrandom()*wY;

		int NN = NMAX*(0.5+0.5*myrandom());									//How many points to add to current neighborhood
		if (NN==0) NN=1;													//We want at least one point

		float R_nb = R_max*(0.3 + myrandom()*0.7);							//Radius of current neighborhood: around R_max
		for(int i=0;!ready && i<NN;++i)										//Generate current neighborhood:
		{
			float alpha  = myrandom()*2*M_PI;								//Random point in current neighborhood (random angle [0,2*M_PI], random radius [0,R_max])
			float radius = myrandom()*R_nb;

			int II = ii + radius*sin(alpha);
			int JJ = jj + radius*cos(alpha);
			if (II<1 || JJ<1 || II>wX-2 || JJ>wY-2) continue;				//Be sure not to generate points on image border (simplifies many calculations later)

			Point2d& np = cloud->points[ngen];
			np = Point2d(II,JJ);
			float val = myrandom();
			cloud->point_scalars[ngen] = val;
			if (cloud->point_scalars_min>val) cloud->point_scalars_min = val;
			if (cloud->point_scalars_max<val) cloud->point_scalars_max = val;

			cloud->min_p.x = std::min(cloud->min_p.x,np.x);
			cloud->min_p.y = std::min(cloud->min_p.y,np.y);
			cloud->max_p.x = std::max(cloud->max_p.x,np.x);
			cloud->max_p.y = std::max(cloud->max_p.y,np.y);

			++ngen;
			ready = ngen == NP;
		}
	}

	for(int i=0;i<NP;++i)
	  for(int j=i;j<NP;++j)
	  {
		float val = myrandom();
		(*cloud->distmatrix)(i,j) = val;
		(*cloud->distmatrix)(j,i) = val;
	  }
}
