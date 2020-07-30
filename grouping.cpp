#include "include/grouping.h"
#include "include/pointcloud.h"
#include "include/sparsematrix.h"
#include "include/fullmatrix.h"
#include <map>


using namespace std;

Grouping::~Grouping()
{
}


//-------  RadialGrouping  -----------------------------


void RadialGrouping::finePoints(int pid,const RadialGrouping* root,PointGroup& fpts) const
{																				//Get all fine-points, at level 'root_cloud', whcih got grouped as 'pid' in this
	if (finer && this!=root)													//Should recurse deeper?
	{
		const PointGroup& pg = point_groups[pid];	
		for(PointGroup::const_iterator it=pg.begin();it!=pg.end();++it)
		{
		  finer->finePoints(*it,root,fpts);
		}	
	}
	else																		//No recursion - report point at this level
		fpts.insert(pid);	
}

RadialGrouping::~RadialGrouping()
{
	delete coarser->cloud;														//We own 'coarser', so we should delete its contents too
	delete coarser;
}

int RadialGrouping::size() const
{ return cloud->size(); }																


RadialGrouping* RadialGrouping::coarsen(float user_avgdist)						//Construct coarser version of this, and add it to this
{
	int NP = cloud->points.size();
	vector<bool> visited(NP);
	vector<int>  nbs;
	
	delete coarser;																//Erase any coarser version of this, since we're gonna replace it
	PointCloud* coarser_cloud = new PointCloud(cloud->fboSize);					//Create coarse point-cloud, to be filled in
	coarser = new RadialGrouping(coarser_cloud);								//Create new coarse level
	
	coarser->finer = this;														//Link coarser level with this
	coarser_cloud->min_p.x = coarser_cloud->min_p.y = 1.0e8;
	coarser_cloud->max_p.x = coarser_cloud->max_p.y = -1.0e8;
	
	const
	vector<Point2d>&      crtp = cloud->points;									//The points at this level		
	vector<Point2d>&      newp = coarser_cloud->points;							//The new coarse points to create
	vector<PointGroup>&     pg = coarser->point_groups;							//The mapping coarse->fine points to create

	if (user_avgdist<0)															//No hint given for avgdist: Use our own estimation
		user_avgdist = cloud->avgdist;
	
	for(int i=0;i<NP;++i)														//Coarsen all points of this:
	{
		if (visited[i]) continue;												//Point already in a group, done
		
		pg.push_back(PointGroup());												//Make new group
		PointGroup& grp = pg[pg.size()-1];
		Point2d center;

		cloud->searchR(crtp[i],user_avgdist,200,nbs);							//Get all fine points in radius eps to current point i
		for(int i=0,NN=nbs.size();i<NN;++i)
		{
		   int nb = nbs[i];
		   if (visited[nb]) continue;											//Point already in a group, done
		   
		   visited[nb] = true;
		   grp.insert(nb);														//Add fine point to current group
		   center += crtp[nb];													//Update group center
		}	
		
		center /= grp.size();													//This is the final group center
		newp.push_back(center);													//...which is a new coarse point
		coarser_cloud->min_p.x = std::min(coarser_cloud->min_p.x,center.x);
		coarser_cloud->min_p.y = std::min(coarser_cloud->min_p.y,center.y);
		coarser_cloud->max_p.x = std::max(coarser_cloud->max_p.x,center.x);
		coarser_cloud->max_p.y = std::max(coarser_cloud->max_p.y,center.y);
	}
	
	
	int CP = coarser_cloud->points.size();
	vector<float>& news = coarser_cloud->point_scalars;							//Scalars for coarse points:
	vector<float>& crts = cloud->point_scalars;
	news.resize(CP);
	float smin = 1.0e+8, smax = -1.0e+8;
	for(int i=0;i<CP;++i)
	{
	   float& label = news[i];
	   label = 1.0e+8;
	   const PointGroup& grp = pg[i];
	   for(PointGroup::const_iterator it=grp.begin();it!=grp.end();++it)
	      label = std::min(label,crts[*it]);
	   smin  = std::min(smin,label);
	   smax  = std::max(smax,label);
	}
	coarser_cloud->point_scalars_min = smin;
	coarser_cloud->point_scalars_max = smax;
	
	vector<vector<float>*>& newa = coarser_cloud->attributes;					//Coarsen attributes:
	vector<vector<float>*>& crta = cloud->attributes;
	int ND = crta.size();
	newa.resize(ND);
	coarser_cloud->attributes_min.resize(ND);
	coarser_cloud->attributes_max.resize(ND);
	for(int i=0;i<ND;++i)
	{
	   newa[i] = new vector<float>(CP);
	   coarser_cloud->attributes_min[i] =  1.0e6;
	   coarser_cloud->attributes_max[i] = -1.0e6;
	}	   
	   	   
	vector<float> attr(ND);   
	for(int i=0;i<CP;++i)
	{
		for(int j=0;j<ND;++j) attr[j]=0;
		const PointGroup& pgi = pg[i];
		
		for(PointGroup::const_iterator it=pgi.begin();it!=pgi.end();++it)
		{
			int pid = *it;
			for(int j=0;j<ND;++j)
			   attr[j] += (*crta[j])[pid];
		}

		for(int j=0;j<ND;++j) 
		{
		  attr[j] /= pgi.size();
		  (*newa[j])[i] = attr[j];
		  coarser_cloud->attributes_min[j] = std::min(coarser_cloud->attributes_min[j],attr[j]);
		  coarser_cloud->attributes_max[j] = std::max(coarser_cloud->attributes_max[j],attr[j]);
		}
	}
	

	const 
	PointCloud::DistMatrix* crtdm = cloud->distmatrix;
	PointCloud::DistMatrix* cdm   = new PointCloud::DistMatrix(CP);				//Coarsen distance matrix:		
	coarser_cloud->distmatrix = cdm;
	
	for(int i=0;i<CP;++i)														//Aggregate coarse-distances from fine ones, based on groups"
	  for(int j=0;j<i;++j)
	  {
		float val = 0;
		for(PointGroup::const_iterator ii=pg[i].begin();ii!=pg[i].end();++ii)
		  for(PointGroup::const_iterator jj=pg[j].begin();jj!=pg[j].end();++jj)
		     val += (*crtdm)(*ii,*jj);
		val /= (pg[i].size()*pg[j].size());										//Distance (error) between groups = average of point-to-point distance over the groups
		
		(*cdm)(i,j) = val;
		(*cdm)(j,i) = val;
	  }

	cdm->minmax();																//Compute range of finer-scale distance matrix	

	cout<<"Coarsened: "<<NP<<"->"<<CP<<" points, with radius="<<user_avgdist<<". Error matrix: ["<<cdm->min()<<","<<cdm->max()<<"]"<<endl;

	coarser_cloud->initEnd();													//Finalize point cloud creation, once all points are added		

	return coarser;																//Return new (coarser) grouping just created
}


void RadialGrouping::group(int gid,PointGroup& g)								//Impl inh
{
	finePoints(gid,0,g);
}


//------   StronglyConnectedGrouping  ----------------------------------

void StronglyConnectedGrouping::build()
{
	int NP = cloud->points.size();	
	
	point_groups.clear();											//Erase any possibly existing contents of this
	
	
	multimap<float,int> order;
	
	for(int i=0;i<NP;++i)											//i: site i-1 at current pixel
	{
		float dist;
		cloud->closest(i,dist);
		order.insert(make_pair(dist,i));
	}

	vector<int> visited(NP);										//Group that i-th point is assigned to, or -1 if not yet assigned
	for(int i=0;i<NP;++i) visited[i] = -1;
	vector<float> avgdist;											//Average interpoint distance in each group

	for(multimap<float,int>::const_iterator it=order.begin();it!=order.end();++it)
	{
		int i = it->second;											//Visit points in increasing distance order to their closest neighbor
		if (visited[i]!=-1) continue;								//Don't visit a point twice (should not happen anyways)
	
		float dmin;													//Find distance and identity of closest neighbor
		int   j  = cloud->closest(i,dmin);
		int   Cj = visited[j];
		
		if (Cj==-1)													//Closest point is not in a group.
		{															//Since we (i,j) are the closest points now, we can form our own group
		   PointGroup pg;
		   pg.insert(i);
		   pg.insert(j);
		   int C = point_groups.size();								//Id of the group
		   visited[i] = C;											//Mark i,j as being in this group
		   visited[j] = C;
		   point_groups.push_back(pg);								//One more group
		   avgdist.push_back(dmin);									//Store average-distance in this group (so far, has 2 points only)
		}
		else														//Closest point is in a group: See if we also can join that group
		{
		   float dj = avgdist[Cj];		
		   if (dmin < 10 || dmin < 1.3*dj)										//We're reasonably close to the group w.r.t. how compact the group is
		   {
			  int N = point_groups[Cj].size();						//Update the average group distance
		      point_groups[Cj].insert(j);							//Add ourselves to the group
			  visited[j] = Cj;
			  avgdist[Cj] = ((N-1)*avgdist[Cj]+dmin)/N;
		   }
		}
	}
	
	cout<<"Groups, pass 1: "<<point_groups.size()<<endl;
	

	float tol = 1.2;	
	for(int iter=0;iter<8;++iter,tol*=1)
	{
	int NG = point_groups.size();
	vector<bool> remove(NG);
	for(int i=0;i<NG;++i) remove[i] = false;
	
	for(int gi=0;gi<NG;++gi)
	{
	   float dmin;	
	   cout<<"Checking "<<gi<<" out of "<<NG<<", pass "<<iter<<endl;
	   int gj = closest(gi,visited,dmin);
	   if (gj==-1) continue;
	   
	   PointGroup& Gi = point_groups[gi];
	   PointGroup& Gj = point_groups[gj]; 
	   
	   bool merge = dmin < 10 || (dmin < tol*avgdist[gi] && dmin < tol*avgdist[gj]);
	   
	   if (!merge) continue;
	   
	   avgdist[gj] = (avgdist[gi]*Gi.size()+avgdist[gj]*Gj.size())/(Gi.size()+Gj.size());
	   for(PointGroup::const_iterator it=Gi.begin();it!=Gi.end();++it)
	   {
		 visited[*it] = gj;
		 Gj.insert(*it);
	   }
	   remove[gi] = true;
	   cout<<"Merging "<<gi<<" -> "<<gj<<endl;
	}
	
	vector<PointGroup> npg;
	vector<float> avgd;
	for(int i=0;i<NP;++i) visited[i]=-1;

	for(int i=0;i<NG;++i)
	   if (!remove[i])
	   {
		  int sz = npg.size();
		  for(PointGroup::const_iterator it=point_groups[i].begin();it!=point_groups[i].end();++it)
		     visited[*it] = sz;

	      npg.push_back(point_groups[i]);
		  avgd.push_back(avgdist[i]);
	   }	
	   
	point_groups = npg;   
	avgdist = avgd;
	
	
	
	cout<<"Groups, pass 2: "<<point_groups.size()<<endl;
	}
}



int StronglyConnectedGrouping::closest(int gid,const vector<int>& visited,float& dmin)
{
	PointGroup& g = point_groups[gid];						//The group we're computing the distance from

    dmin = 1.0e6;
	int   gmin = -1;

															//Scan all points of this group:
	for(PointGroup::const_iterator it=g.begin();it!=g.end();++it)
	{
	    int pid = *it;	
		const Point2d& pt = cloud->points[pid];
	   	
	    const PointCloud::TrisOfPoint& fan = cloud->point2tris[pid];
	    for(PointCloud::TrisOfPoint::const_iterator fti=fan.begin();fti!=fan.end();++fti)
	    {													//Find all Delaunay neighbor-points	of current point,
	      int ft = *fti;									//which are NOT in this group, and record the closest group thereby
		  const Triangle& tr = cloud->triangles[ft];

		  for(int idx=0;idx<3;++idx)
		  {
		    int p  = tr(idx);
			int gp = visited[p];
		    if (gp==gid || gp==-1) continue;				//Didn't find a different neighbor-group, nothing to do
			if (gp<gid) continue;
			
		    float dist = pt.dist(cloud->points[p]);
			if (dist>dmin) continue;
			
			//cout<<"For grp "<<gid<<": found closer: "<<gp<<", dist "<<dist<<" between "<<pid<<","<<p<<" visited[]="<<visited[pid]<<","<<visited[p]<<endl;

			dmin = dist;
			gmin = gp;
			
		  }	
	    }
	}	
	
	//cout<<"Closest to "<<gid<<": "<<gmin<<", dist "<<dmin<<endl;

	return gmin;
}



