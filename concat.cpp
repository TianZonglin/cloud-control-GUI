#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include "include/point2d.h"


using namespace std;



vector<Point2d> points1,points2;
vector<float>   labels1,labels2;
vector<float>   dist;


bool read2D(const string& f,vector<Point2d>& points,vector<float>& labels,Point2d& min,Point2d& max);
void computeDist(const vector<Point2d>& p1,const vector<Point2d>& p2,vector<float>& dist);
bool writeError(const string&,vector<Point2d>& points1,vector<Point2d>& points2,const vector<float>&);
bool writePoints(const string& f,const vector<Point2d>& p1,const vector<Point2d>& p2,const vector<float>& l1,const vector<float>& l2);


int main(int argc,char** argv)
{
   --argc; ++argv;
   
   if (argc!=3)
   {
	  cout<<"Error: #arguments "<<argc<<", expected 2"<<endl;
	  return 1;
   }
   
   string file  = argv[0]; --argc; ++argv;
   string proj1 = argv[0]; --argc; ++argv;
   string proj2 = argv[0]; --argc; ++argv;
   
   
   string f2d_1 = file;													//2-dimensional projection file
   f2d_1 += "."; f2d_1 += proj1;			
   f2d_1 += ".2d";
	
   string f2d_2 = file;													//2-dimensional projection file
   f2d_2 += "."; f2d_2 += proj2;			
   f2d_2 += ".2d";
   	

   Point2d min1,max1,min2,max2;
   read2D(f2d_1,points1,labels1,min1,max1);	
   read2D(f2d_2,points2,labels2,min2,max2);	
   if (points1.size()!=points2.size())
   {
      cout<<"Error: set 1 has "<<points1.size()<<" points, set 2 has "<<points2.size()<<" points"<<endl;
	  return 1;
   }	  

   const int NP = points1.size();
   Point2d t = min1-min2;												//1. Translate set 2 to origin of set 1
   for(int i=0;i<NP;++i)
      points2[i] += t;

   float sx = (max1.x-min1.x)/(max2.x-min2.x);							//2. Scale set 2 to match largest-dim of set 1
   float sy = (max1.y-min1.y)/(max2.y-min2.y);
   float  s = min(sx,sy);
   for(int i=0;i<NP;++i)
   {
      points2[i].x = min1.x+(points2[i].x-min1.x)*sx;
      points2[i].y = min1.y+(points2[i].y-min1.y)*sy;
   }

   float maxl=0;
   for(int i=0;i<NP;++i)
      maxl = std::max(maxl,labels1[i]);

   cout<<"Max label in "<<f2d_1<<": "<<int(maxl)<<endl;

   for(int i=0;i<NP;++i)
   {
     labels2[i] += maxl;
   }

   	     
   string f2d_12 = file+"."+proj1+"."+proj2+".2d";
   writePoints(f2d_12,points1,points2,labels1,labels2);
   
   computeDist(points1,points2,dist);									//3. Compute distance of corresponding points in sets 1,2
   
   string ferr_12 = file+"."+proj1+"."+proj2+".err";					//4. Write 'error' distance-matrix based on (3) above
   writeError(ferr_12,points1,points2,dist);

   return 0;
}




bool writePoints(const string& f,const vector<Point2d>& p1,const vector<Point2d>& p2,const vector<float>& l1,const vector<float>& l2)
{
	FILE* fp = fopen(f.c_str(),"w");								//1. Read 2D projections:
	if (!fp)
	{
		cout<<"Error: Cannot open "<<f<<endl;
		return false;
	}	
	
	const int NP = points1.size();

	fprintf(fp,"DY\n");	
	fprintf(fp,"%d\n",2*NP);	
	fprintf(fp,"2\n\n");	
	
	for(int i=0;i<NP;++i)
	{
	   fprintf(fp,"%d;%f;%f;%f\n",i,p1[i].x,p1[i].y,l1[i]);
	}
	for(int i=0;i<NP;++i)
	{
	   fprintf(fp,"%d;%f;%f;%f\n",i+NP,p2[i].x,p2[i].y,l2[i]);
	}
	
	fclose(fp);
	return true;
}



bool read2D(const string& f,vector<Point2d>& points,vector<float>& labels,Point2d& min_p,Point2d& max_p)
{
    char line[1024];

	FILE* fp = fopen(f.c_str(),"r");								//1. Read 2D projections:
	if (!fp)
	{
		cout<<"Error: Cannot open "<<f<<endl;
		return false;
	}	


	fgets(line,1024,fp);											//Skip first line 'DY'
	int NP;
	fscanf(fp,"%d",&NP);											//Get #points in file
	int dim;
	fscanf(fp,"%d",&dim);											//Get point dimensions (should be 2)
	if (dim!=2)
	{
	   cout<<"Warning: 2D projection dimension="<<dim<<", expected 2"<<endl;
	}

	points.resize(NP);
	labels.resize(NP);
	min_p = Point2d(1.0e+6,1.0e+6);	
	max_p = Point2d(-1.0e+6,-1.0e+6);	
	for(int i=0;i<NP;++i)											//Read all 2D point projections:
	{
		Point2d& p = points[i];
		fscanf(fp,"%*[^;];%f;%f;%f",&p.x,&p.y,&labels[i]);			//REMARK: apparently, first item (point-ID) can be a string..
		min_p = min_p.min(p);
		max_p = max_p.max(p);
	}
	fclose(fp);

	return true;
}


void computeDist(const vector<Point2d>& p1,const vector<Point2d>& p2,vector<float>& dist)
{
	const int N = p1.size();
	dist.resize(N);
	
	float dist_max=0;
	for(int i=0;i<N;++i)
	{
	   float d = p1[i].dist(p2[i]);
	   dist[i] = d;
	   dist_max = max(d,dist_max); 	
	}
	
	if (dist_max<1.0e-6) dist_max=1;
	
	for(int i=0;i<N;++i)
	{
	   dist[i] /= dist_max;
	}
	
	cout<<"dist_max "<<dist_max<<endl;
}


bool writeError(const string& f,vector<Point2d>& points1,vector<Point2d>& points2,const vector<float>& dist)
{
	FILE* fp = fopen(f.c_str(),"w");								//1. Read 2D projections:
	if (!fp)
	{
		cout<<"Error: Cannot open "<<f<<endl;
		return false;
	}	
	
	const int NP = points1.size();
	fprintf(fp,"%d\n",2*NP);
	
	for(int row=0;row<2*NP;++row)
	{
		for(int col=0;col<=row;++col)
		{
			if (row<NP || col>=NP) 
			   fprintf(fp,"0.0");
			else 
			{
			   int r = row-NP;
			   if (r!=col)
				  fprintf(fp,"0.0");
			   else
			   {
			      fprintf(fp,"%f",dist[r]);
			   }	  
			}  
			
			if (col<row) fprintf(fp,";");
		}
		fprintf(fp,"\n");
	}
	

	fclose(fp);
	return true;
}


