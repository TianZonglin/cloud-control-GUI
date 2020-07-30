#pragma once
#include <vector>
#include "include/hashwrap.h"

class	PointCloud;	


class	Grouping									//Baseclass for a grouping of points from a cloud.
{													//No restrictions are placed on groups 
public:												//(e.g. can be overlapping, or not covering the whole group)

typedef hash_set<int>	PointGroup;


						Grouping(PointCloud* c):cloud(c) {}
virtual				   ~Grouping();
virtual	int				size() const =0;			//Get #groups in this
virtual void			group(int,PointGroup&) =0;	//Get point-ids of i-th group into arg2

PointCloud*				cloud;						//Point cloud this grouping works on. Not owned by this.
};




class SimpleGrouping : public Grouping				//Simple grouping specialization: Stores groups explicitly
{
public:

						SimpleGrouping(PointCloud* c): Grouping(c) {}
int						size() const				//Impl inh
						{ return point_groups.size(); }								
void					resize(int sz) 
						{ point_groups.resize(sz); }
void					group(int i,PointGroup& g)	//Impl inh
						{ g = point_groups[i]; }
PointGroup&				group(int i)				//Accessor, mainly for writing
						{ return point_groups[i]; }
						
protected:

std::vector<PointGroup> point_groups;				//All groups of this grouping, explicitly stored. They refer to points in 'cloud'
};





class	RadialGrouping : public Grouping			//Specialized grouping that uses a hierarchical distance-based grouping.
{
public:
						RadialGrouping(PointCloud* c): Grouping(c),finer(0),coarser(0) {}
					   ~RadialGrouping();	
RadialGrouping*			coarsen(float avgdist=-1);								//Coarsen this point-cloud up to a user-given, or otherwise estimated reasonable level-of-detail
int						size() const;				//Impl inh
void					group(int i,PointGroup& g);	//Impl inh

RadialGrouping*			finer;													//Finer-scale cloud that this approximates (if any), else 0
RadialGrouping*			coarser;												//Coarser-scale cloud that simplifies this (if any, else 0. Owned by this

private:

void					finePoints(int pid,const RadialGrouping* root,PointGroup& fpts) const;					
																				//Get all points, at 'root' level, coarsened to 'pid' in this cloud

std::vector<PointGroup> point_groups;											//All groups of this grouping, in terms of point-ids in 'finer'
};



class StronglyConnectedGrouping : public SimpleGrouping
{
public:
						StronglyConnectedGrouping(PointCloud* c): SimpleGrouping(c) {}

void					build();												//Create this grouping

int						closest(int gid,const std::vector<int>& visited,float& dmin);
};

