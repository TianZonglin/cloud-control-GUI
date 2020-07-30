#pragma once

#include <math.h>

class Point3d
{
public:
  union {
    float data[3];
    struct { float x,y,z; };			//coordinates
  };
  float		n[3];			//normal
  float		value;			//some context-dependent metric value (typically, the importance for skel points)
      
			Point3d(float xx=0, float yy=0, float zz=0) : x(xx), y(yy), z(zz), value(0) 
			{ n[0]=n[1]=n[2]=0; }
			Point3d(const float* p): x(p[0]),y(p[1]),z(p[2]), value(0)
			{ n[0]=n[1]=n[2]=0; }

 void       flip() {x=-x; y=-y; z=-z; }
 void		flipNormal() { n[0]=-n[0]; n[1]=-n[1]; n[2]=-n[2]; }
 float		dist(const Point3d &p) const { return sqrtf( (x-p.x)*(x-p.x) + (y-p.y)*(y-p.y) + (z-p.z)*(z-p.z) ); }
 float		dist2(const Point3d &p) const { return (x-p.x)*(x-p.x) + (y-p.y)*(y-p.y) + (z-p.z)*(z-p.z); }
 float		norm() const { return sqrtf(x*x+y*y+z*z); }
 float		norm2() const { return x*x+y*y+z*z; } 
 float*		Normal() { return n; }
 const
 float*		Normal() const { return n; }
 float		dot(const Point3d &v1) const { return x*v1.x + y*v1.y + z*v1.z; }
 float      dot(const float* v) const  { return x*v[0] + y*v[1] + z*v[2]; }
 bool		nonzero() const { return x!=0 || y!=0 || z!=0; }

 float& operator[](int idx) { return data[idx]; }

 operator	float*() { return &x; }
 operator	const float*() const { return &x; }
 bool       isApprox(const Point3d& p,float tol) const { return fabs(x-p.x)+fabs(y-p.y)+fabs(z-p.z)<tol; }

 Point3d    cross(const Point3d &v2) const { return Point3d(y*v2.z-v2.y*z, v2.x*z-x*v2.z, x*v2.y-v2.x*y); } 
 Point3d	operator-(const Point3d &p) const { return Point3d(x-p.x, y-p.y, z-p.z); }
 Point3d    operator-() const { return Point3d(-x,-y,-z); }	
 Point3d	operator+(const Point3d &p) const { return Point3d(x+p.x, y+p.y, z+p.z); }
 Point3d&   operator+=(const Point3d &p) { x+=p.x; y+=p.y; z+=p.z; return *this; }
 Point3d&   operator-=(const Point3d &p) { x-=p.x; y-=p.y; z-=p.z; return *this;}
 Point3d	operator*(float k) const { return Point3d(k*x,k*y,k*z); }
 Point3d&   operator*=(float k) { x*=k; y*=k; z*=k; return *this; }
 
 bool operator<(const Point3d &p) const { return x<p.x && y<p.y && z<p.z; };

 Point3d	operator/(float k) const { return Point3d(x/k, y/k, z/k); }
 Point3d&   operator/=(float k) { x/=k; y/=k; z/=k; return *this; }

 float		normalize() { float n = norm(); if (n<1.0e-6) n=1; x/=n; y/=n; z/=n; return n; }
 
 Point3d(const Point3d &p): x(p.x),y(p.y),z(p.z),value(p.value)
			{ n[0]=p.n[0]; n[1]=p.n[1]; n[2]=p.n[2]; }

 Point3d&	operator=(const Point3d &p) 
			{ 
			  if (&p==this) return *this;
			  x=p.x; y=p.y; z=p.z; value=p.value; 
			  n[0]=p.n[0]; n[1]=p.n[1]; n[2]=p.n[2];
			  return *this;
			}
			
 Point3d&	operator=(const float* p)
			{
			  x=p[0]; y=p[1]; z=p[2];
			  return *this;
			}

void		interpolate(const Point3d& a, const Point3d& b,float t)
			{ float tt=1-t; x = a.x*t+b.x*tt; y = a.y*t+b.y*tt; z = a.z*t+b.z*tt; }
			
 float		distPlane(const Point3d &normal, const Point3d &ppoint)
 {
   return (x-ppoint.x)*normal.x + (y-ppoint.y)*normal.y + (z-ppoint.z)*normal.z;
 }

 //set this to a vector perpendicular to vec 
 void setPerp(const Point3d &vec) 
 {
   float a = fabsf(vec.x), b=fabsf(vec.y), c=fabsf(vec.z);
    
   x = y = z = 0.0f;
     
   if(a<b && a<c) x = 1.0f;
   else if(b<c) y = 1.0f;
   else z = 1.0f;

   *this = cross(vec);
 }
};      

//-------------------------------


//Some free functions follow; these are general point-related geometric utilities

inline void sub(float* res,const float* a,const float* b)
{
  res[0]=a[0]-b[0];
  res[1]=a[1]-b[1];
  res[2]=a[2]-b[2];
}


inline void cross(const float *v1, const float *v2, float *r)
{
  r[0]=v1[1]*v2[2]-v2[1]*v1[2];
  r[1]=v2[0]*v1[2]-v1[0]*v2[2];
  r[2]=v1[0]*v2[1]-v2[0]*v1[1];
}

inline float norm(const float *v) 
{ return sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }

inline float dot(const float *v1, const float *v2)
{ return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]; }

inline float normalize(float *v)
{
  register float len = 1e-7f + norm(v);
  v[0]/=len; v[1]/=len; v[2]/=len;
  return len;
}

