#pragma once

#include <vector>
#include <math.h>


struct Point2d					//General-purpose 2D point
{
	float x,y;

	Point2d():x(0),y(0) {}
	Point2d(float xx,float yy): x(xx), y(yy) {}
	bool valid() const { return x!=0 && y!=0; }
	
	Point2d operator+(const Point2d& p) const
	{  return Point2d(x+p.x,y+p.y); }

	Point2d operator-(const Point2d& p) const
	{  return Point2d(x-p.x,y-p.y); }

	Point2d operator*(float t) const
	{  return Point2d(x*t,y*t); }
	
	Point2d& operator-=(const Point2d& p)
	{  x -= p.x; y -= p.y; return *this; }
	
	Point2d& operator+=(const Point2d& p)
	{  x += p.x; y += p.y; return *this; }

	float norm() const
	{ return sqrt(x*x+y*y); }		

	float norm2() const
	{ return x*x+y*y; }		
	
	void normalize()
	{
		float nrm = norm();
		if (nrm>1.0e-6) 
		{ x /= nrm; y /= nrm; }
	}
	
	Point2d& operator*=(float t)
	{  x *= t; y *= t; return *this; }

	Point2d& operator*=(const Point2d& p)
	{  x *= p.x; y *= p.y; return *this; }

	Point2d& operator/(const Point2d& p)
	{  x /= p.x; y /= p.y; return *this; }


	Point2d& operator/=(float t)
	{  x /= t; y /= t; return *this; }

	Point2d& operator=(const Point2d& p)
	{  x = p.x; y = p.y; return *this; }
	
	Point2d interp(const Point2d& p2,float t)
	{
		return Point2d((1-t)*x+t*p2.x,(1-t)*y+t*p2.y);
	}
	
	float dot(const Point2d& p) const
	{   return x*p.x + y*p.y;  }
	
	float distance2line(const Point2d& p0, const Point2d& p1) const
	{
		const Point2d l = p1-p0;
		float dp = (*this-p0).dot(l);
		float d  = (*this-p0).norm2() - dp*dp/l.norm2();
		return sqrt(d);
	}
	
	Point2d closestOnLine(const Point2d& p0, const Point2d& p1) const
	{
		const Point2d l = p1-p0;		
		float dp = (*this-p0).dot(l) / l.norm2();
		Point2d q = l*dp+p0;
		return q;
	}
	
	float dist(const Point2d& p) const
	{  return sqrt((x-p.x)*(x-p.x)+(y-p.y)*(y-p.y));  }
	
	float dist2(const Point2d& p) const
	{  return (x-p.x)*(x-p.x)+(y-p.y)*(y-p.y);  }

	float cross(const Point2d& v2) const { return x*v2.y-v2.x*y; } 


	bool inTriangle(const Point2d& a,const Point2d& b,const Point2d& c) const
	{
	  Point2d v1 = a-*this, v2 = b-*this, v3 = c-*this;
	
	  float c1 = v1.cross(v2);
	  float c2 = v2.cross(v3);
	  float c3 = v3.cross(v1);

	  return ((c1>0 && c2>0 && c3>0) || (c1<0 && c2<0 && c3<0));
	}

	Point2d min(const Point2d& rhs) const
	{ 
	  Point2d ret;
	  ret.x = (x<rhs.x)? x:rhs.x;
	  ret.y = (y<rhs.y)? y:rhs.y;
	  return ret;
	}

	Point2d max(const Point2d& rhs) const
	{ 
	  Point2d ret;
	  ret.x = (x>rhs.x)? x:rhs.x;
	  ret.y = (y>rhs.y)? y:rhs.y;
	  return ret;
	}
	
	static float	edgeAngle(const Point2d&,const Point2d&);
	static float	angle(const Point2d&);
	static Point2d	center(const Point2d&,const Point2d&,const Point2d&);			//inscribed circle center for given triangle
};

	

typedef std::vector<Point2d> PointSet;


