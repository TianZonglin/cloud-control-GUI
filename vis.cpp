#include "include/vis.h"
#include "include/scalarimage.h"
#include "include/pointcloud.h"
#include "include/simplepointcloud.h"
#include "include/graph.h"
#include "include/fullmatrix.h"
#include "include/orderedmatrix.h"
#include "include/sortederrmatrix.h"
#include "include/io.h"
#include "include/grouping.h"
#include "include/vgrouping.h"
#include "include/myrandom.h"
#include "include/skelft.h"
#include "include/gdrawingcloud.h"
#include "include/cpubundling.h"
#include "include/cudawrapper.h"

#include "include/glwrapper.h"
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <math.h>
#include  "include/explanatorymap.h"
#include "include/config.h"

using namespace std;


RadialGrouping* visual_clustering = 0;				//!!To be added to some clean interface
Grouping* labelg = 0;


enum
{
		UI_SCALE_DOWN=1,
		UI_SCALE_UP,
		UI_TEX_INTERP,
		UI_SHOW_POINTS,
		UI_SHOW_DELAUNAY,
		UI_COLOR_MODE,
		UI_SHOW_GROUPS,
		UI_SHOW_MAPTYPE,
		UI_GROUP_FINER,
		UI_GROUP_COARSER,
		UI_SHOW_BRUSH,
		UI_SHOW_SELECTION,
		UI_SHOW_CLOSEST_SEL,
		UI_RECOMPUTE_CUSHIONS,
		UI_SKEL_CUSHIONS,
		UI_CUSHION_THICKNESS,
		UI_CUSHION_ALPHA,
		UI_MAP_ALPHA,
		UI_POINTS_ALPHA,
		UI_FALSEPOS_DISTWEIGHT,
		UI_FALSEPOS_RANGE,
		UI_FALSENEG_RANGE,
		UI_AGGREGATE_ERR_RANGE,
		UI_CUSHION_COLORING,
		UI_CUSHION_STYLE,
		UI_POINT_RADIUS,
		UI_SHEPARD_AVERAGING,
		UI_CUSHION_OPENING,
		UI_SHOW_BUNDLES,
		UI_BUNDLE_ITERATIONS,
		UI_BUNDLE_KERNEL,
		UI_BUNDLE_EDGERES,
		UI_BUNDLE_SMOOTH,
		UI_BUNDLE_DENS_ESTIM,
		UI_BUNDLE_CPU_GPU,
		UI_BUNDLE_SAVE,
		UI_BUNDLE_FALSE_POS,
		UI_SCATTER_X_AXIS,
		UI_SCATTER_Y_AXIS,
		UI_QUIT
};

static float	 relaxation = 0;
static float	 dir_separation = 0;
static bool		 shading = true;
static float	 shading_radius = 3;

////static int		 show_particles = 0;					//Visualization options: Eventually all these have to move in Display
static int		 show_delaunay  = 0;
static int		 color_mode     = 1;
static int		 show_groups    = 0;
static int		 show_bundles	= 0;
////static int		 show_maptype   = 0;
////static int		 show_brush	    = 1;
static int		 show_selection = 1;
static int		 show_closest_sel = 1;
static float	 map_alpha		= 1;
////static float	 points_alpha	= 1;
static int		 skeleton_cushions = 0;
static float	 cushion_threshold = 0.35;
static float	 cushion_shading_thickness = 30;
static float	 point_size     = 2;
static float	 cushion_alpha  = 1;
static int		 cushion_coloring = 1;
static int		 cushion_style  = 0;
////static float	 shepard_averaging = 4;
////static int		 false_positive_distweight = 10;
////static float	 false_positive_range = 0;
////static float	 false_negative_range = 0;
////static float	 aggregate_error_range = 0;
static float	 opening_threshold = 20;
static float	 frac_false_positives_bundle = 0.01;	//Fraction of false positive edges (from total N^2 edges) to show in bundling
static int		 density_estimation = 0;
static int		 gpu_bundling = 1;
static GLUI_StaticText* ui_false_negatives_bundling_size = 0;
static GLUI_StaticText* ui_selection_size = 0;
static float	 MAX_SELECT_DIST = 20;					//Max distance for interactively selecting points (screen-space)
static float	 interpolation_level = 0;				//Mouse-controlled interpolation level between the current Cartesian and current distribution view
static int		 interpolation_dir   = 0;				//Interpolation direction control: -1=left motion, 1=right motion, 0=uninitialized
static int		 scatter_x_axis = 2;
static int		 scatter_y_axis = 2;


static float  bundling_relaxation = 0;




static vector<hash_set<int> >							//For each group, the triangle-ids in the Delaunay triangulation of the cloud
				 xtris;									//whose vertices belong to this group


Display*		Display::instance = 0;					//!!!Move following vars to class...
VisualGrouping	visual_groups;
float*			skel_dt;
float*			dt_param;
CPUBundling*	bund = 0;								//Bundling engine (for various graphs)
Graph*			false_negatives_graph = 0;				//Graph linking a point with its false negatives, for all points
GraphDrawingCloud*
				false_negatives_bundling = 0;			//Drawing of above
GraphDrawingCloud*
				gdrawing_final = 0;			//Drawing of above
GraphDrawingCloud xx;



Grouping::PointGroup selection;							//Selected points in the visualization
set<int>		  selected_group;						//Id of selected label-group (if any was selected), else -1
SimplePointCloud* pcloud_projection;
SimplePointCloud* pcloud_distribution;
SimplePointCloud *current_view,*next_view;





//------------------------------------------------------------------------------------------------


void draw_image(Display& dpy,int tex_id)
{
	float t = 1-interpolation_level;

	setTexture(tex_id,dpy.tex_interp);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glColor4f(1,1,1,map_alpha*t);
	glBegin(GL_QUADS);
	glTexCoord2f(0,0); glVertex2f(0.0, 0.0);
	glTexCoord2f(1,0); glVertex2f(dpy.winSize, 0.0);
	glTexCoord2f(1,1); glVertex2f(dpy.winSize, dpy.winSize);
	glTexCoord2f(0,1); glVertex2f(0.0, dpy.winSize);
	glEnd();
	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_2D);
}


void Display::computeDimensionRankMap() {
    
//    ExplanatoryMap *expMap = new ExplanatoryMap();
//    expMap->cloud = this->cloud;
//    expMap->buildTopRanked();
     
    ExplanatoryMap *expMap = new ExplanatoryMap();
    cloud->dimrank_topdims.clear();
    //printf("========b========");
    ////if (!bundlingHistogram.empty() && show_bundles) {
    ////    vector<int> topDims;
    ////    for (int i = 0; i < 8; i++) {
    ////        topDims.push_back(bundlingHistogram[i].dimId);
    ////    }
    ////   cloud->dimrank_topdims = topDims;
    ////}    
    
    cloud->filterRankings(true);    
    //printf("========c========");
    expMap->cloud = cloud;
    expMap->buildTopRankedNew();

	//////////////////////////// MARKER

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,tex_dimranking);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,imgSize,imgSize,0,GL_RGBA,GL_FLOAT,expMap->image);
    glDisable(GL_TEXTURE_2D);

    delete expMap->image;    
    delete expMap;
    
    return;
}



void draw_dimension_ranking(Display& dpy) {
	/*
    if (draw_dimrank_heatmap) {
        draw_image(dpy,dpy.tex_dimranking_heat);       
                
        int minSet = INT_MAX;
        int maxSet = 0;      
        for (int i = 0; i < dpy.cloud->point_dimset.size(); i++) {
            if (dpy.cloud->point_dimset[i].rankings.size() > maxSet)
                maxSet = dpy.cloud->point_dimset[i].rankings.size();
            if (dpy.cloud->point_dimset[i].rankings.size() < minSet)
                minSet = dpy.cloud->point_dimset[i].rankings.size();          
        }
        dpy.drawColormap(CMAP_HEATED, minSet, maxSet, Point2d(dpy.imgSize + 30, 10));        
        return;
    } else if (draw_dimrank_setmap) {
            draw_image(dpy,dpy.tex_dimranking_set);
            VisualHistogram rankingSetHistogram = dpy.computeRankingSetHistogram();
            dpy.drawHistogram(rankingSetHistogram);
            return;
    }
    */

 



    draw_image(dpy,dpy.tex_dimranking);
 
    //VisualHistogram rankingHistogram = dpy.computeRankingHistogram();
    //rankingHistogram.title = "Global dimension ranking";
    //dpy.drawHistogram(rankingHistogram);    
}



void draw_false_positives(Display& dpy)
{
	draw_image(dpy,dpy.tex_color);
}

void draw_dt(Display& dpy)
{
	draw_image(dpy,dpy.tex_dt);
}

void draw_false_negatives(Display& dpy)
{
		if (selection.size()==0) return;								//No selected points? Nothing to do
		draw_image(dpy,dpy.tex_false_negatives);
}

void draw_aggregate_error(Display& dpy)
{
	draw_image(dpy,dpy.tex_aggregate);
}


void draw_label_mixing(Display& dpy)
{
	draw_image(dpy,dpy.tex_mixing);
}

void draw_error_distribution(Display& dpy)
{
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPointSize(point_size*dpy.scale);
	glColor4f(0,0,0,points_alpha);

	glBegin(GL_POINTS);
	for(int i=0;i<pcloud_distribution->size();++i)
	{
	  const Point2d& p = pcloud_distribution->points[i];
	  const Color& col = pcloud_distribution->colors[i];
	  glColor4f(col.r,col.g,col.b,points_alpha);
	  glVertex2f(p.x,p.y);
	}
	glEnd();

	glDisable(GL_BLEND);
	glPointSize(1);
}



void Display::drawSelection()
{
	const SimplePointCloud& pc1 = *current_view;
	const SimplePointCloud& pc2 = *next_view;
	float t  = interpolation_level;									//Interpolate: 0=current_view, 1=next_view

	if (show_selection)
	{
		glEnable(GL_POINT_SMOOTH);
		glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		for(int i=0;i<2;++i)											//1. Draw selected points as marked:
		{
			float size  = (2-i)*point_size*scale;
			glPointSize(size);
			if (i==0)
				glColor3f(1,0,0);
			else
				glColor3f(1,1,1);
			glBegin(GL_POINTS);
			for(Grouping::PointGroup::const_iterator it=selection.begin();it!=selection.end();++it)
			{
				int i = *it;
				const Point2d& p1 = pc1.points[i];
				const Point2d& p2 = pc2.points[i];
				Point2d p = p1*(1-t) + p2*t;
				glVertex2f(p);
			}
			glEnd();
		}
		glDisable(GL_BLEND);
		glPointSize(1);
	}

	if (show_closest_sel)
	{
		if (selected_point_id!=-1)										//2. Draw closest-selected-point:
		{
			glShadeModel(GL_FLAT);
			for(int i=0;i<2;++i)										//Draw a nice outlined cursor at 'selected_point_id'
			{
				float color = (i==0)?1:0;
				float width = (i==0)?3:1;
				glColor3f(color,color,color);							//Draw a crosshairs at selected point
				glLineWidth(width);
				const Point2d& sp1 = pc1.points[selected_point_id];
				const Point2d& sp2 = pc2.points[selected_point_id];
				Point2d sp = sp1*(1-t) + sp2*t;
				float del = 0.025*winSize;
				glBegin(GL_LINES);
				glVertex2f(sp.x-del,sp.y);
				glVertex2f(sp.x+del,sp.y);
				glVertex2f(sp.x,sp.y-del);
				glVertex2f(sp.x,sp.y+del);
				glEnd();
			}
			glLineWidth(1);
		}
	}
}






Display::Display(int glutWin_,
	       int winSize_,						//Graphics window size; can be anything smaller, equal, or bigger to texture size
				 PointCloud* cloud_,				//Point cloud
				 int argc,char** argv):				//Arguments: needed for GLUT
				 imgSize(winSize_),					//WARNING: Here, we assume the image is square and pow(2)
				 cloud(cloud_),
				 winSize(winSize_), 
				 scale(1),
				 transX(0),transY(0),
				 isLeftMouseActive(false),isRightMouseActive(false),
				 oldMouseX(0),oldMouseY(0),
			     tex_interp(true),
				 closest_point(-1),
				 selected_point_id(-1)
{
    printf("\n----Display::Display(...)\n");
	glutInitWindowSize(winSize_, winSize_);								//Graphics system initialization: This must occur in a very specific order!
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA);
	glutInit(&argc, argv);	
	//  1. first initialize GLUT
  	glutWin_ = glutCreateWindow("Graph bundling");	  
  	//  2. then create one main window, which initializes OpenGL
	glewInit();                                         
	//  3. then initialize GLEW



	instance = this;
    canvasSize = winSize_;
    
    point_influence_radius = point_influence_radius * canvasSize / 1024.0f;
    shepard_averaging = shepard_averaging * canvasSize / 1024.0f;

 
	cudaMallocHost((void**)&cushion_param,winSize*winSize*sizeof(float));
	cudaMallocHost((void**)&cushion_dt,winSize*winSize*sizeof(float));
	splat_img = new float[winSize*winSize];
	dt_param  = new float[winSize*winSize];
	skel_dt   = new float[winSize*winSize];
	image     = new ScalarImage(winSize,winSize);

	pcloud_projection   = new SimplePointCloud(cloud->size());
	pcloud_distribution = new SimplePointCloud(cloud->size());
	current_view = pcloud_projection;
	next_view    = pcloud_distribution;
 
    glutWin = glutWin_;
    glutDisplayFunc(display_cb);
	//glutMouseFunc(mouse_cb);
	//glutKeyboardFunc(keyboard_cb);
    //glutMotionFunc(motion_cb);
	//glutPassiveMotionFunc(passivemotion_cb);

	GLuint texture[20];								//Textures for displaying various objects
    glGenTextures(20,texture);						//Generate all required textures
	tex_sites = texture[0];
	tex_dt    = texture[1];
	tex_color = texture[2];
	tex_mask  = texture[3];
	tex_splat = texture[4];
	tex_point_density = texture[5];
	tex_mixing = texture[6];
	tex_aggregate = texture[7];
	tex_false_negatives = texture[8];
	tex_framebuffer_lum = texture[9];
	tex_density = texture[10];
	tex_framebuffer_rgba = texture[11];
	tex_dimranking = texture[12];

	GLuint framebuffers[2];
	glGenFramebuffers(2,framebuffers);			//Make two offscreen framebuffers: high-res luminance one (for accurate splatting) and low-res RGBA one (for all other ops)
	framebuffer_lum  = framebuffers[0];
	framebuffer_rgba = framebuffers[1];
 
	glBindFramebuffer(GL_FRAMEBUFFER,framebuffer_lum);
	glBindTexture(GL_TEXTURE_2D,tex_framebuffer_lum);
	//Make empty texture
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F_ARB,winSize,winSize,0,GL_LUMINANCE,GL_UNSIGNED_BYTE,0);		 
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	//Connect texture to framebuffer
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT0_EXT,GL_TEXTURE_2D,tex_framebuffer_lum,0);	 

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,framebuffer_rgba);
	glBindTexture(GL_TEXTURE_2D,tex_framebuffer_rgba);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,winSize,winSize,0,GL_RGBA,GL_UNSIGNED_BYTE,0);							//Make empty texture
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT0_EXT,GL_TEXTURE_2D,tex_framebuffer_rgba,0);//Connect texture to framebuffer

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);		//Make sure we draw in the onscreen buffer, after all above stuff

    colorMap.load(current_cmap);
    visual_clustering = new RadialGrouping(cloud);
    labelg = cloud->groupByLabel();  

    cloud->dimensionRankCentroid();
    cout << "Computing dimension rankings..." << endl;    
    cloud->dimensionRank(dimrank_radius);
	cout << "then..." << endl;    
	//visual_groups.init(labelg);
	//visual_groups.cushion_shading_thickness = cushion_shading_thickness;



	//bund = new CPUBundling(winSize);				//Create bundling engine; we'll use it for several graph bundling tasks in this class
	//bund->initEdgeProfile(CPUBundling::PROFILE_HOURGLASS);
	//bund->verbose = false;
	//density_estimation = bund->density_estimation;

    glPixelStorei(GL_UNPACK_ALIGNMENT,1);			//Initialize some state-related stuff in this
    glPixelStorei(GL_PACK_ALIGNMENT,1);
	//makeSplat(1024,2.0);								//Create a reasonably fine-grained radial distance splat
	//makeDensitySplat(1024);							//REMARK: Obsolete?

	//cloud->computeAggregateError(aggregate_error_range);
	//cloud->computeLabelMixing();

	//////////computeTriangleMap();							//Requires a RGBA framebuffer


	////// appear error
	/////////////computeFalseNegativesGraph();/**/

 

	//////////computeBundles();
	/////generateTexture();
    cout << "\n. then..." << endl; 
/*
glEnable(GL_TEXTURE_2D);
	typedef unsigned char BYTE;
	BYTE*  tex  = new BYTE[imgSize * imgSize * 3];		
 			//Normalize 'image' automatically or vs user-specified range
    for (int i = 0; i < imgSize; ++i){												//Generate visualization texture
        for (int j = 0; j < imgSize; ++j)
		{
			int   id   = j * imgSize + i;
            tex[id * 3 + 0] = 255;
            tex[id * 3 + 1] = 255;
            tex[id * 3 + 2] = 0;
			//if(tex[id * 3 + 0]!=255&&tex[id * 3 + 1]!=255&&tex[id * 3 + 2]!=255)
			// printf("%d,%d,%d\t",tex[id * 3 + 0],tex[id * 3 + 1],tex[id * 3 + 2]);
		}
	}
 printf("%d,%d,%d\t",tex[0],tex[1],tex[2]);
	glBindTexture(GL_TEXTURE_2D, tex_color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imgSize, imgSize, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
	glDisable(GL_TEXTURE_2D);
    delete[] tex;
*/










	//////////computeAllCushions();
	//////////computeGroupMeshes(visual_clustering);
	////////computeLabelMixing();
	//computePointColors(show_particles);
	//computeDistribution();
 
	computeDimensionRankMap();
	//computeDimensionRankSetMap();

 

	char buf[128];
	GLUI_Panel *pan,*pan2;								//Construct GUI:
	GLUI_Scrollbar* scr;
	glui = GLUI_Master.create_glui("Projections Wizard");
/* 
	GLUI_Rollout* ui_map = glui->add_rollout("Map",false);					//1. Panel "Map":
	pan = glui->add_panel_to_panel(ui_map,"Show what");
	GLUI_RadioGroup* ui_maptype = new GLUI_RadioGroup(pan,&show_maptype,UI_SHOW_MAPTYPE,control_cb);
	new GLUI_RadioButton(ui_maptype,"False positives (all points)");
	new GLUI_RadioButton(ui_maptype,"False negatives (selection)");
	new GLUI_RadioButton(ui_maptype,"Aggregate error (all points)");
	new GLUI_RadioButton(ui_maptype,"Label mixing");
	new GLUI_RadioButton(ui_maptype,"Points' DT");
	new GLUI_RadioButton(ui_maptype,"Error distribution (all points)");
	new GLUI_RadioButton(ui_maptype,"Nothing");
	pan = glui->add_panel_to_panel(ui_map,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"False positive smoothing");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&false_positive_distweight,UI_FALSEPOS_DISTWEIGHT,control_cb);
	scr->set_int_limits(0,50);
	pan = glui->add_panel_to_panel(ui_map,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"False positive range");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&false_positive_range,UI_FALSEPOS_RANGE,control_cb);
	scr->set_float_limits(0,1);
	pan = glui->add_panel_to_panel(ui_map,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"False negative range");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&false_negative_range,UI_FALSENEG_RANGE,control_cb);
	scr->set_float_limits(0,1);
	pan = glui->add_panel_to_panel(ui_map,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Aggregate error range");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&aggregate_error_range,UI_AGGREGATE_ERR_RANGE,control_cb);
	scr->set_float_limits(0,1);
	pan = glui->add_panel_to_panel(ui_map,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Map alpha");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&map_alpha,UI_MAP_ALPHA,control_cb);
	scr->set_float_limits(0,1);


	GLUI_Rollout* ui_visuals = glui->add_rollout("Point cloud",false);	//2. Panel "Visual settings":
	pan = glui->add_panel_to_panel(ui_visuals,"Cloud triangulation");
	GLUI_RadioGroup* ui_delaunay = new GLUI_RadioGroup(pan,&show_delaunay,UI_SHOW_DELAUNAY,control_cb);
	new GLUI_RadioButton(ui_delaunay,"Don't show");
	new GLUI_RadioButton(ui_delaunay,"Black");
	new GLUI_RadioButton(ui_delaunay,"Colored");

	pan = glui->add_panel_to_panel(ui_visuals,"Cloud drawing");
	GLUI_Listbox* ui_lb = new GLUI_Listbox(pan,"Color",&show_particles,UI_SHOW_POINTS,control_cb);
	ui_lb->add_item(0,"Nothing");
	ui_lb->add_item(1,"Black");
	ui_lb->add_item(2,"Label");
	ui_lb->add_item(3,"False negatives (selection)");
	ui_lb->add_item(4,"Aggregate error");
	ui_lb->add_item(5,"Aggregate FP error");
	ui_lb->add_item(6,"Aggregate FN error");
	for(int i=0;i<cloud->attributes.size();++i)
	{
	  char buf[120];
	  sprintf(buf,"Attribute %d",i);
	  ui_lb->add_item(i+6,buf);
	}

	GLUI_Rollout* ui_scatter = glui->add_rollout("Scatterplot",false);	//2. Panel "Scatterplot":
	ui_lb = new GLUI_Listbox(ui_scatter,"X axis",&scatter_x_axis,UI_SCATTER_X_AXIS,control_cb);
	ui_lb->add_item(2,"Label");
	ui_lb->add_item(3,"False negatives (selection)");
	ui_lb->add_item(4,"Aggregate error");
	ui_lb->add_item(5,"Aggregate FP error");
	ui_lb->add_item(6,"Aggregate FN error");
	for(int i=0;i<cloud->attributes.size();++i)
	{
	  char buf[120];
	  sprintf(buf,"Attribute %d",i);
	  ui_lb->add_item(i+6,buf);
	}

	ui_lb = new GLUI_Listbox(ui_scatter,"Y axis",&scatter_y_axis,UI_SCATTER_Y_AXIS,control_cb);
	ui_lb->add_item(2,"Label");
	ui_lb->add_item(3,"False negatives (selection)");
	ui_lb->add_item(4,"Aggregate error");
	ui_lb->add_item(5,"Aggregate FP error");
	ui_lb->add_item(6,"Aggregate FN error");
	for(int i=0;i<cloud->attributes.size();++i)
	{
	  char buf[120];
	  sprintf(buf,"Attribute %d",i);
	  ui_lb->add_item(i+6,buf);
	}

	pan = glui->add_panel_to_panel(ui_visuals,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Points alpha");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&points_alpha,UI_POINTS_ALPHA,control_cb);
	scr->set_float_limits(0,1);



	GLUI_Rollout* ui_grps = glui->add_rollout("Point groups",false);		//3. Panel "Point groups":
	pan = glui->add_panel_to_panel(ui_grps,"Display groups");
	GLUI_RadioGroup* ui_groups = new GLUI_RadioGroup(pan,&show_groups,UI_SHOW_GROUPS,control_cb);
	new GLUI_RadioButton(ui_groups,"Nothing");
	new GLUI_RadioButton(ui_groups,"Visual groups");
	new GLUI_RadioButton(ui_groups,"Label groups");
	new GLUI_RadioButton(ui_groups,"Mesh-based groups");
	new GLUI_RadioButton(ui_groups,"Point density");

	pan = glui->add_panel_to_panel(ui_grps,"Cushion style");
	GLUI_RadioGroup* ui_cushion_style = new GLUI_RadioGroup(pan,&cushion_style,UI_CUSHION_STYLE,control_cb);
	new GLUI_RadioButton(ui_cushion_style,"Border");
	new GLUI_RadioButton(ui_cushion_style,"Full");

	new GLUI_Checkbox(ui_grps,"Skeleton cushions", &skeleton_cushions, UI_RECOMPUTE_CUSHIONS, control_cb);
	new GLUI_Checkbox(ui_grps,"Cushion coloring", &cushion_coloring, UI_CUSHION_COLORING, control_cb);
	pan = glui->add_panel_to_panel(ui_grps,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Cushion border");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&cushion_shading_thickness,UI_CUSHION_THICKNESS,control_cb);
	scr->set_float_limits(1,50);
	pan = glui->add_panel_to_panel(ui_grps,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Cushion thickness");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&cushion_threshold,UI_RECOMPUTE_CUSHIONS,control_cb);
	scr->set_float_limits(0,1);
	pan = glui->add_panel_to_panel(ui_grps,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Cushion smoothness");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&opening_threshold,UI_CUSHION_OPENING,control_cb);
	scr->set_float_limits(1,100);
	pan = glui->add_panel_to_panel(ui_grps,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Cushion alpha");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&cushion_alpha,UI_CUSHION_ALPHA,control_cb);
	scr->set_float_limits(0,1);

	new GLUI_Button(ui_grps,"Recompute cushions",UI_RECOMPUTE_CUSHIONS,control_cb);

	GLUI_Rollout* ui_stats = glui->add_rollout("Statistics",false);			//4. Panel "Statistics":
	pan = glui->add_panel_to_panel(ui_stats,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Points");
	glui->add_column_to_panel(pan,false);
	sprintf(buf,"%d",cloud->size());
	new GLUI_StaticText(pan,buf);
	pan = glui->add_panel_to_panel(ui_stats,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Dimensions");
	glui->add_column_to_panel(pan,false);
	sprintf(buf,"%d",cloud->dimensions());
	new GLUI_StaticText(pan,buf);
	pan = glui->add_panel_to_panel(ui_stats,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Labels");
	glui->add_column_to_panel(pan,false);
	sprintf(buf,"%d",int(cloud->numLabels()));
	new GLUI_StaticText(pan,buf);
	pan = glui->add_panel_to_panel(ui_stats,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"False-positive bundles");
	glui->add_column_to_panel(pan,false);
	sprintf(buf,"%d",false_negatives_bundling->numEdges());
	ui_false_negatives_bundling_size = new GLUI_StaticText(pan,buf);
	pan = glui->add_panel_to_panel(ui_stats,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Selected points");
	glui->add_column_to_panel(pan,false);
	sprintf(buf,"%d",(int)selection.size());
	ui_selection_size = new GLUI_StaticText(pan,buf);


	//!!!Show data for selected-point

	GLUI_Rollout* ui_settings = glui->add_rollout("General settings",false);//5. Panel "Settings":
	new GLUI_Checkbox(ui_settings, "Interpolate tex", &tex_interp, UI_TEX_INTERP, control_cb);
	new GLUI_Checkbox(ui_settings, "Colormapping", &color_mode, UI_COLOR_MODE, control_cb);
	new GLUI_Checkbox(ui_settings, "Brush", &show_brush, UI_SHOW_BRUSH, control_cb);
	new GLUI_Checkbox(ui_settings, "Selection", &show_selection, UI_SHOW_SELECTION, control_cb);
	new GLUI_Checkbox(ui_settings, "Closest selected point", &show_closest_sel, UI_SHOW_CLOSEST_SEL, control_cb);
	pan = glui->add_panel_to_panel(ui_settings,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Point influence radius");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&cloud->avgdist,UI_POINT_RADIUS,control_cb);
	scr->set_float_limits(2,75);
	pan = glui->add_panel_to_panel(ui_settings,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Shepard smoothing");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&shepard_averaging,UI_SHEPARD_AVERAGING,control_cb);
	scr->set_int_limits(1,50);

    new GLUI_Button(glui,"Quit",UI_QUIT,control_cb);						//5. "Quit" button:


	glui->add_column(true);													//--------------------------------------------------------

	GLUI_Rollout* ui_bundling = glui->add_rollout("Bundling",false);		//4. Panel "Bundling":
	pan = glui->add_panel_to_panel(ui_bundling,"False negatives (all)");
	new GLUI_StaticText(pan,"Show what");
	GLUI_RadioGroup* ui_bundles = new GLUI_RadioGroup(pan,&show_bundles,UI_SHOW_BUNDLES,control_cb);
	new GLUI_RadioButton(ui_bundles,"Nothing");
	new GLUI_RadioButton(ui_bundles,"Bundles");

	pan = glui->add_panel_to_panel(ui_bundling,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"False negatives to show");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&frac_false_positives_bundle,UI_BUNDLE_FALSE_POS,control_cb);
	scr->set_float_limits(0,1);
	pan2 = glui->add_panel_to_panel(ui_bundling,"General options");
	pan = glui->add_panel_to_panel(pan2,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Iterations");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->niter,UI_BUNDLE_ITERATIONS,control_cb);
	scr->set_int_limits(0,30);
	pan = glui->add_panel_to_panel(pan2,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Kernel size");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->h,UI_BUNDLE_KERNEL,control_cb);
	scr->set_float_limits(3,40);
	pan = glui->add_panel_to_panel(pan2,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Smoothing");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->lambda,UI_BUNDLE_SMOOTH,control_cb);
	scr->set_float_limits(0,1);
	pan = glui->add_panel_to_panel(pan2,"",GLUI_PANEL_NONE);
	new GLUI_StaticText(pan,"Edge sampling");
	glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->spl,UI_BUNDLE_EDGERES,control_cb);
	scr->set_float_limits(3,50);
	pan = glui->add_panel_to_panel(pan2,"Density estimation");
	GLUI_RadioGroup* ui_dens_estim = new GLUI_RadioGroup(pan,&density_estimation,UI_BUNDLE_DENS_ESTIM,control_cb);
	new GLUI_RadioButton(ui_dens_estim,"Exact");
	new GLUI_RadioButton(ui_dens_estim,"Fast");
	new GLUI_Checkbox(pan2,"GPU method", &gpu_bundling,UI_BUNDLE_CPU_GPU,control_cb);
	new GLUI_Button(ui_bundling,"Save",UI_BUNDLE_SAVE,control_cb);
*/	
	glui->set_main_gfx_window(glutWin);		 							//Link GLUI with GLUT (seems needed)
}




Display::~Display()															//Dtor
{																			//!!Not ready: much more needs to be deleted
	delete false_negatives_bundling;
	delete false_negatives_graph;
	delete pcloud_projection;
	delete pcloud_distribution;
	delete bund;
	delete dt_param;
	delete skel_dt;
	delete image;
	cudaFreeHost(cushion_param);
	cudaFreeHost(cushion_dt);
	delete[] splat_img;
}


void Display::computeFalseNegativesGraph()									//Build graph of most important 'frac_false_positives_bundle'-percent of false-negatives
{

    if (cloud->distmatrix == NULL)
        return;
    
    int NP = cloud->size();	
    typedef multimap<float,Graph::Edge> ValueEdges;
    ValueEdges sorted;
    const PointCloud::DistMatrix& dm = *cloud->distmatrix;	
                                                                                                                                                   //1. Sort all FPs (so we can next select most important ones):
    if (selection.size())													//1.1. If we have a selection, we only show edges going to/from selected points:
    {
       for(Grouping::PointGroup::const_iterator it=selection.begin();it!=selection.end();++it)
       {																	//Start with one endpoint IN selection:
              int i = *it;
              const PointCloud::DistMatrix::Row& row = dm[i];
          for(int j=0;j<NP;++j)
          {
            float err = row[j];
                if (err<0) continue;											//Skip false positives, we don't want those ones
                    if (selection.find(j)!=selection.end()) continue;				//Skip edges ENDING in selection (since we want only edges linking selection with outside)			
            sorted.insert(make_pair(err,make_pair(i,j)));
          }
       }
    }
    else
    for(int i=0;i<NP;++i)													//1.2. Nothing selected: show all FPs in entire dataset:
    {
       const PointCloud::DistMatrix::Row& row = dm[i];
       for(int j=i+1;j<NP;++j)
       {
         float err = row[j];
             if (err<0) continue;												//Skip false positives, we don't want those ones
         sorted.insert(make_pair(err,make_pair(i,j)));
       }
    }

    int Nmax = frac_false_positives_bundle * sorted.size();					//Retain only the most important 'frac_false_positives_bundle' FPs

    int   I = 0;															//2. Construct the graph (as a sparse adj-matrix)
    float max_err = sorted.rbegin()->first, min_err;
    delete false_negatives_graph;											//
    false_negatives_graph = new Graph(NP);
    for(ValueEdges::const_reverse_iterator it=sorted.rbegin();it!=sorted.rend() && I<Nmax;++it,++I)
    {
        min_err = it->first;
            float norm_err = min_err/max_err;
            const Graph::Edge& edge = it->second;	
            (*false_negatives_graph)(edge.first,edge.second) = min_err;
            (*false_negatives_graph)(edge.second,edge.first) = min_err;
    }

    delete false_negatives_bundling;    
    false_negatives_bundling = new GraphDrawingCloud();						//3. Construct a graph drawing for 'false_negatives_graph'	
    
	////////////appear error
	false_negatives_bundling->build(false_negatives_graph,cloud);			//  

    if (ui_false_negatives_bundling_size)									//3. Update Statistics UI (if any)
    {
            char buf[128];
            sprintf(buf,"%d",false_negatives_bundling->numEdges());
            ui_false_negatives_bundling_size->set_text(buf);

            cout<<"FP bundles: error in ["<<min_err<<","<<max_err<<"]"<<endl;
    } 	
}


void Display::computeGroupMeshes(Grouping* g)
{
	xtris.resize(g->size());

	for(int i=0;i<g->size();++i)
	{
	  hash_set<int>& tris = xtris[i];
 	  tris.clear();

	  Grouping::PointGroup pg;
	  g->group(i,pg);

	  for(Grouping::PointGroup::const_iterator it=pg.begin();it!=pg.end();++it)
	  {
	    int pid = *it;										//Get the triangle-fan (fine-scale) of fine-scale point

	    const PointCloud::TrisOfPoint& fan = cloud->point2tris[pid];
	    for(PointCloud::TrisOfPoint::const_iterator fti=fan.begin();fti!=fan.end();++fti)
	    {													//Select all triangles in fan with vertices only in 'pg'
	      int ft = *fti;
		  const Triangle& tr = cloud->triangles[ft];
		  if (pg.find(tr(0))==pg.end()) continue;
		  if (pg.find(tr(1))==pg.end()) continue;
		  if (pg.find(tr(2))==pg.end()) continue;
		  tris.insert(ft);
	    }
	  }
	}
}


void Display::drawMap()											//Display one of the various dense maps:
{
	printf("\nGrouping* PointCloud::groupByLabel()\n",show_maptype);
	switch (6)  
	//error here type=7 that I cancelled. becasue it's a global parameter so I changed it in
	//error several places, that might have a position I didn't change back！！！ and there is no 
	//error error that's strange.
	{
	case 0: draw_false_positives(*this);	break;
	case 1: draw_false_negatives(*this);	break;
	case 2: draw_aggregate_error(*this);	break;
	case 3: draw_label_mixing(*this);		break;
	case 4: draw_dt(*this);					break;
	case 5: draw_error_distribution(*this); break;
	case 6: draw_dimension_ranking(*this); 	break;
	//case 7: draw_set(*this); 	break;
	}
}


void Display::drawGroups()
{/*
	if (show_groups==3)											//Show mesh-based groups
	{
		drawGroupMeshes();
		return;
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

	if (show_groups==2)											//Show image-based label groups:
	{
	    for(int i=0;i<labelg->size();++i)
		{
		   if (selected_group.find(i)==selected_group.end()) continue;



		   VisualGrouping::Cushion* c = visual_groups.getCushion(i);
		   visual_groups.draw(c,cushion_alpha,cushion_coloring,tex_interp);
		}
	}

	if (show_groups==1)											//Show visual grouping of all points in the cloud:
	{
	   VisualGrouping::Cushion* c = visual_groups.getCushion(-1);
	   visual_groups.draw(c,cushion_alpha,false,tex_interp);
	}

	if (show_groups==4)											//Show the density map of all points in the cloud:
	{
		glColor4f(0,0,0,cushion_alpha);
		setTexture(tex_point_density,tex_interp);
		glBegin(GL_QUADS);
		glTexCoord2f(0,0); glVertex2f(0.0, 0.0);
		glTexCoord2f(1,0); glVertex2f(winSize, 0.0);
		glTexCoord2f(1,1); glVertex2f(winSize, winSize);
		glTexCoord2f(0,1); glVertex2f(0.0, winSize);
		glEnd();
	}

	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_2D);*/
}



void Display::startOffscreen(int ncomponents)
{
	if (ncomponents==1)
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,framebuffer_lum);		//Draw in the offscreen buffer:
	else
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,framebuffer_rgba);

	glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);									//Start clean

    glViewport(0,0,winSize,winSize);
    glMatrixMode(GL_PROJECTION);									//Setup projection matrix
	glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0,winSize,0,winSize);
    glMatrixMode(GL_MODELVIEW);										//Must reset viewing transformations, since we next
	glPushMatrix();													//want to draw all points as they are in the cloud (i.e., untransformed)
    glLoadIdentity();
}

void Display::endOffscreen()
{
    glMatrixMode(GL_PROJECTION);									//Restore whatever transformations we had originally
	glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);						//Redirect drawing in the onscreen buffer
}


void Display::computeTriangleMap()
{
	startOffscreen(4);
	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_2D);
	glShadeModel(GL_FLAT);

	glBegin(GL_TRIANGLES);											//Render all triangles, and encode 1+triangle-id in the rendered RGB color.
	for(int i=0,NT=cloud->triangles.size();i<NT;++i)				//This creates a color buffer where 0 means no triangle, and else we can
	{																//find the triangle-id from the color
		unsigned int id = i+1;
		unsigned int i0 = id & 255; id >>= 8;
		unsigned int i1 = id & 255; id >>= 8;
		unsigned int i2 = id & 255;
		glColor3ub(i2,i1,i0);

		const Triangle& tr = cloud->triangles[i];
		glVertex2f(cloud->points[tr(0)]);
		glVertex2f(cloud->points[tr(1)]);
		glVertex2f(cloud->points[tr(2)]);
	}
	glEnd();

	glReadPixels(0,0,winSize,winSize,GL_RGBA,GL_UNSIGNED_BYTE,cloud->buff_triangle_id);
	unsigned char* buff = (unsigned char*)cloud->buff_triangle_id;	//We read the data as RGBA (for speed). Postprocess it so that in each
	for(int i=0;i<winSize*winSize;++i)								//buff_triangle_id[] element we truly have the triangle-id+1 at that pixel, or 0.
	{
		unsigned int v2 = *buff++;
		unsigned int v1 = *buff++;
		unsigned int v0 = *buff++;
		buff++;
		cloud->buff_triangle_id[i] = (v2<<16) + (v1<<8) + v0;
	}

	endOffscreen();
}




void Display::drawGroupMeshes()
{
	int NG = xtris.size();
	for(int i=0;i<NG;++i)
	{
	    float r,g,b;
		float2rgb(float(i)/NG,r,g,b,color_mode);
		glColor3f(r,g,b);

		glBegin(GL_TRIANGLES);
		hash_set<int>& tris = xtris[i];
		for(hash_set<int>::const_iterator it=tris.begin();it!=tris.end();++it)
		{
			int tid = *it;
			const Triangle& t = cloud->triangles[tid];
			glVertex2f(cloud->points[t(0)]);
			glVertex2f(cloud->points[t(1)]);
			glVertex2f(cloud->points[t(2)]);
		}
		glEnd();

	}
}


void Display::displayCb()
{
    glClearColor(1,1,1,1);											//Reset main GL state to defaults
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glViewport(0,0,winSize,winSize);
    glMatrixMode(GL_PROJECTION);									//Setup projection matrix
    glLoadIdentity();
    gluOrtho2D(0,winSize,0,winSize);
    glMatrixMode(GL_MODELVIEW);										//Setup modelview matrix
    glLoadIdentity();
    glScalef(scale,scale,1);
    glTranslatef(transX,transY,0);


	//1. Draw the current map-visualization on the background
	drawMap();

	//2. Draw the groups atop of this visualization, appropriately blended
	////if (show_groups!=0) drawGroups();

	//3. Draw the Delaunay edges
	////if (show_delaunay!=0) drawDelaunay();

	//4. Draw the points in the cloud
    //////////////drawPoints();

	//5. Draw any bundles that may be there
	////if (show_bundles!=0) drawBundles();

	//6. Draw any selection that may be there
	drawSelection();

	//7. Draw the interactive brush
	drawBrush();

    glutSwapBuffers();												// All done
}



void Display::drawDelaunay()
{
		glColor3f(0,0,0);

		glBegin(GL_LINES);
		for(int i=0;i<cloud->points.size();++i)
		{
			const Point2d& pi = cloud->points[i];
			const PointCloud::EdgeMatrix::Row& row = (*cloud->sorted_edges)(i);
			for(PointCloud::EdgeMatrix::Row::const_iterator it = row.begin();it!=row.end();++it)
			{
				int     j = it->pid;
				const Point2d& pj = cloud->points[j];

				if (show_delaunay==2)
				{
					float r,g,b;
					float v = (*cloud->distmatrix)(i,j);
					float2rgb(v,r,g,b,color_mode);
					glColor3f(r,g,b);
				}

				glVertex2f(pi);
				glVertex2f(pj);
			}
		}
		glEnd();
}

void Display::drawBundles()
{
	switch (show_bundles)
	{
	case 1:
		false_negatives_bundling->draw(false_negative_range);
		break;
	}
}



void Display::computePointColors(int color_attr)
{
	const vector<float>* attrs=0; float sm,rng;

	switch(color_attr)
	{
	case 0:
	case 1:
		break;
	case 2:
		{
		  sm    = cloud->point_scalars_min;
		  rng   = cloud->point_scalars_max - sm;
		  attrs = &cloud->point_scalars;
		}
		break;
	case 3:
		{
		  sm	= 0;
		  rng	= 1;
		  attrs = &cloud->false_negative_error;
		}
		break;
	case 4:
		{
		  sm    = 0;
		  rng   = 1;
		  attrs = &cloud->aggregate_error;
		}
		break;
	case 5:
		{
		  sm    = 0;
		  rng   = 1;
		  attrs = &cloud->aggregate_fp_error;
		}
		break;
	case 6:
		{
		  sm    = 0;
		  rng   = 1;
		  attrs = &cloud->aggregate_fn_error;
		}
		break;
	default:
		{
		  int a = color_attr-7;
		  sm    = cloud->attributes_min[a];
		  rng   = cloud->attributes_max[a]-sm;
		  attrs = cloud->attributes[a];
		}
	}

	if (rng<1.0e-6) rng = 1;

	for(int i=0;i<cloud->points.size();++i)
	{
		const Point2d& p = cloud->points[i];
		Color col(0,0,0);
		if (attrs)
		{
		  float v = ((*attrs)[i]-sm)/rng;
		  float2rgb(v,col.r,col.g,col.b,color_mode);
		}
		pcloud_projection->points[i] = p;
		pcloud_projection->colors[i] = col;
	}
}



void Display::computeDistribution()
{
	const vector<float> *tx,*ty;
	float mx,my,rx,ry;

	switch(scatter_x_axis)
	{
	case 2:
		mx = cloud->point_scalars_min;
		rx = cloud->point_scalars_max-mx;
		tx = &cloud->point_scalars;
		break;
	case 3:
		mx = 0;
		rx = 1;
		tx = &cloud->aggregate_error;
		break;
	case 4:
		mx = 0;
		rx = 1;
		tx = &cloud->aggregate_fp_error;
		break;
	case 5:
		mx = 0;
		rx = 1;
		tx = &cloud->aggregate_fn_error;
		break;
	}
	if (rx<1.0e-6) rx = 1;

	switch(scatter_y_axis)
	{
	case 2:
		my = cloud->point_scalars_min;
		ry = cloud->point_scalars_max-my;
		ty = &cloud->point_scalars;
		break;
	case 3:
		my = 0;
		ry = 1;
		ty = &cloud->aggregate_error;
		break;
	case 4:
		my = 0;
		ry = 1;
		ty = &cloud->aggregate_fp_error;
		break;
	case 5:
		my = 0;
		ry = 1;
		ty = &cloud->aggregate_fn_error;
		break;
	}
	if (ry<1.0e-6) ry = 1;

	for(int i=0;i<cloud->points.size();++i)
	{
	  Point2d pd;
	  pd.x = winSize*((*tx)[i]-mx)/rx;
	  pd.y = winSize*((*ty)[i]-my)/ry;
	  Color   col = pcloud_projection->colors[i];					//For the moment, use same colors as in the Cartesian projection
	  pcloud_distribution->points[i] = pd;
	  pcloud_distribution->colors[i] = col;
	}
}



void Display::drawPoints()											//Draw points in the cloud
{
    //No point drawing
    if (show_particles==0) 
        return;									

    int currentcolorMapBak = current_cmap;
    colorMap.load(CMAP_HEATED);
    
    
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(point_size*scale);
    glColor4f(cloud_color[0],cloud_color[1],cloud_color[2],points_alpha);

    const vector<float>* attrs=0; float sm,rng;

    if (show_particles==2)                                                      //color by label
    {
        sm    = cloud->point_scalars_min;
        rng   = cloud->point_scalars_max - sm;
        attrs = &cloud->point_scalars;
    }
    else 
        if (show_particles>2)
	{
            int a = show_particles-3;
            sm    = cloud->attributes_min[a];
            rng   = cloud->attributes_max[a]-sm;
            attrs = cloud->attributes[a];
	}

    if (rng<1.0e-6) 
        rng = 1;
    
    Color color;
    glBegin(GL_POINTS);
    for(int i=0;i<cloud->points.size();++i)
    {
        const Point2d& p = cloud->points[i];
        if (attrs)
        {
            float r,g,b;
            float v = ((*attrs)[i]-sm)/rng;
            color = colorMap.getColor(v);
            glColor4f(color.r, color.g, color.b,points_alpha);
        } 
        else
            glColor4f(cloud_color[0],cloud_color[1],cloud_color[2],points_alpha);      
        glVertex2f(p);
    }
    glEnd();

    glDisable(GL_BLEND);
    glPointSize(1);
    
    colorMap.load(currentcolorMapBak);
	
}











void Display::makeSplat(int SZ,float sigma)							//Create a Gaussian splat texture SZ*SZ pixels. Used later for density-map construction
{
	float* img = new float[SZ*SZ];

	const float half = 0.5;
	const float C = SZ/2.0;
	const float C2 = C*C;											//Generate a half-sphere height profile, encode it as a luminance-alpha texture
	const float  k = exp(-sigma);
	for(int i=0,idx=0;i<SZ;++i)										//The height should be normalized in [0..1]; alpha indicates the valid vs nonvalid pixels
		for(int j=0;j<SZ;++j,++idx)
		{
			float x  = i-C+half, y = j-C+half;
			int   r2 = x*x+y*y;
			if (r2>C2)												//Point outside the ball: color irrelevant, and it's transparent
			{
				img[idx] = 0;
			}
			else
			{
				float   D = (exp(-sigma*r2/C2)-k)/(1-k);
				img[idx] = D;
			}
		}

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,tex_splat);
	glTexImage2D(GL_TEXTURE_2D,0,GL_LUMINANCE,SZ,SZ,0,GL_LUMINANCE,GL_FLOAT,img);
	glDisable(GL_TEXTURE_2D);
	delete[] img;
}


void Display::makeDensitySplat(int SZ)								//
{
	float* img = new float[SZ*SZ];

	const float half = 0.5;
	const float C = SZ/2.0;
	const float C2 = C*C;											//Generate a half-sphere height profile, encode it as a luminance-alpha texture
	for(int i=0,idx=0;i<SZ;++i)										//The height should be normalized in [0..1]; alpha indicates the valid vs nonvalid pixels
		for(int j=0;j<SZ;++j,++idx)
		{
			float x  = i-C+half, y = j-C+half;
			int   r2 = x*x+y*y;
			if (r2>C2)												//Point outside the ball: color irrelevant, and it's transparent
			{
				img[idx] = 0;
			}
			else
			{
				float   D = 1-(r2/C2)*(r2/C2);
				img[idx] = D;
			}
		}

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,tex_density);
	glTexImage2D(GL_TEXTURE_2D,0,GL_LUMINANCE32F_ARB,SZ,SZ,0,GL_LUMINANCE,GL_FLOAT,img);
	glDisable(GL_TEXTURE_2D);
	delete[] img;
}



void Display::computeBundles()
{
	false_negatives_bundling->densityMap      = bund->h_densityMap;
	false_negatives_bundling->shadingMap      = bund->h_shadingMap;
	false_negatives_bundling->shading = true;

	bund->setInput(false_negatives_bundling);					//Bundle the graph drawing
	if (gpu_bundling)
		bund->bundleGPU();
	else
		bund->bundleCPU();

	*gdrawing_final = *false_negatives_bundling;								//Don't modify the bundled graph, copy it (because we want to redo postprocessing w/o redoing bundling)


	//if (relaxation || dir_separation)
	gdrawing_final->interpolate(xx,relaxation,dir_separation,1,true);
																	//Relax bundling towards original graph, and also separate edge-directions

	//if (shading || color_mode==GraphDrawing::DENSITY_MAP)			//Compute shading+density of relaxed graph, if we need them
		//bund->computeShading(false_negatives_bundling,shading_radius);

	*false_negatives_bundling = *gdrawing_final;
}



void Display::computeAllCushions()
{
	float* cushions = new float[winSize*winSize];				//1. Compute the visual cushions and cloud's density map
	bool norm = computeCushions(0,cushions);
	visual_groups.setCushion(-1,cushions,norm);

	glBindTexture(GL_TEXTURE_2D,tex_point_density);				//Save the density map (for visual debugging purposes)
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,winSize,winSize,0,GL_ALPHA,GL_FLOAT,splat_img);

	const int NG = labelg->size();								//2. Compute cushions for all groups:
	for(int i=0;i<NG;++i)										//
	{															//
	   Grouping::PointGroup pg;									//
	   labelg->group(i,pg);
	   bool norm = computeCushions(&pg,cushions);
	   visual_groups.setCushion(i,cushions,norm);
	}

	delete[] cushions;

	visual_groups.makeTextures();								//Make all textures for the existing visual cushions
}




bool Display::computeCushions(Grouping::PointGroup* pg,float* output)
{
	startOffscreen(4);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE,GL_ONE);										//We want to accumulate (add) what we draw
	setTexture(tex_splat,true);
	glColor3f(0.1,0.1,0.1);											//Splat must be drawn grey, given that we next read/threshold luminance

	const int winSize2 = winSize*winSize;

	glBegin(GL_QUADS);
	if (!pg)														//Splat the entire cloud:
	for(int i=0,N=cloud->size();i<N;++i)							//1. Compute the fuzzy distance map:
	{
		const Point2d& p = cloud->points[i];
		float rad = cloud->avgdist;									//Radius: What we'd need, is an estimation of the LOCAL inter-point distance..
		if (rad<10) rad=10;											//We don't want too small radii - visually, we cannot distinguish differences at such resolutions
		drawSplat(p,rad);
	}
	else															//Splat a single group:
	{
	  memset(dt_param,0,winSize2*sizeof(float));					//Need to compute DT of only points in _given_ group:
	  for(Grouping::PointGroup::const_iterator it=pg->begin();it!=pg->end();++it)
	  {
		int i = *it;
		const Point2d& p = cloud->points[i];
		float rad = cloud->avgdist;									//Radius: What we'd need, is an estimation of the LOCAL inter-point distance..
		if (rad<10) rad=10;											//We don't want too small radii - visually, we cannot distinguish differences at such resolutions
		drawSplat(p,rad);
		dt_param[int(p.y)*winSize+int(p.x)] = 1;
	  }
	}
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glDisable(GL_BLEND);											//Get the distance map in 'splat_img'
	glReadPixels(0,0,winSize,winSize,GL_LUMINANCE,GL_FLOAT,splat_img);

	float* dt;														//dt = DT of point-set to use (either whole point-cloud or group)
	/*
	if (!pg)
	   dt = cloud->siteDT;
	else															//For groups, we must compute their point-set DT here on the fly:
	{
		skelft2DFT(0,dt_param,0,0,winSize,winSize,winSize);			//Compute FT of the sites
		skelft2DDT(dt_param,0,0,winSize,winSize);					//Compute DT of the sites (from the resident FT)
		dt = dt_param;
	}
	*/

	//!!!
	dt = splat_img;

	skelft2DMakeBoundary(dt,0,0,winSize,winSize,cushion_param,winSize,cushion_threshold,true);
	skelft2DFT(0,cushion_param,0,0,winSize,winSize,winSize);		//1. Threshold 'dt' to obtain a contour roughly around all points
	skelft2DDT(cushion_dt,0,0,winSize,winSize);						//1.1. Compute DT of this contour (for thinning)

	for(int i=0;i<winSize2;++i)										//1.1. Erode the contour with a distance 'opening_threshold' inwards
	   dt_param[i] = (cushion_dt[i]>opening_threshold &&
					  dt[i]>cushion_threshold);						//This removes thin features of the contour (which is good)

	skelft2DFT(0,dt_param,0,0,winSize,winSize,winSize);				//2. Compute DT of the eroded contour (for inflation)
	skelft2DDT(cushion_dt,0,0,winSize,winSize);						//

	for(int i=0;i<winSize2;++i)										//3. Inflate the eroded contour outwards with a distance 'opening_threshold'
	  if (cushion_dt[i]>opening_threshold)							//For this, compute the eroded contour's DT and threshold it by 'opening_threshold'
		 cushion_dt[i]=0;											//The result is a kind of morphological opening of the initial contour
	  else
		 cushion_dt[i] = opening_threshold - cushion_dt[i];

	float length = skelft2DMakeBoundary(cushion_dt,0,0,winSize,winSize,cushion_param,winSize,0.00001,true);
	skelft2DFT(0,cushion_param,0,0,winSize,winSize,winSize);		//4. Finally, compute the inwards-DT of the thinned-and-inflated shape
	skelft2DDT(skel_dt,0,0,winSize,winSize);						//This is simply needed for shading the interior of this shape
	for(int i=0;i<winSize2;++i)
	   cushion_dt[i] = (cushion_dt[i]>0)? skel_dt[i] : 0;

	if (skeleton_cushions)											//Use skeleton-cushion-shading: compute interpolation of shape-to-skeleton
	{																//(via well-known blend formula)
	  skelft2DSkeleton(0,length,30,0,0,winSize,winSize);
	  skel2DSkeletonDT(skel_dt,0,0,winSize,winSize);
	}

	for(int i=0;i<winSize2;++i)
	{
	   float v=-1;													//Outside cushions: set height to -1 (marker for outside)
	   if (cushion_dt[i]>0)											//Inside cushions: copy height from either DT or skeleton-DT interpolation
	   {
		 if (skeleton_cushions)										//Skeleton cushions:
		   v = skel_dt[i];
		 else														//Classical DT cushions:
		 {
  	       v = cushion_dt[i]-1;										//Small correction of DT needed, since 'inside' means DT>0..
		   if (v<0) v=0;
		 }
	   }
	   output[i] = v;												//output[] is always in {-1} U [0,1]
	}

	endOffscreen();
	return skeleton_cushions;										//Skeleton cushions are normalized; DT-cushions are not
}






void Display::generateTexture()
{
	image->interpolateDistMatrix(*cloud,false_positive_distweight);								//Construct the false-positive image from the core data

	const float radius = cloud->avgdist;														//This is the area-of-influence of a point in _2D_ (thus, image-size-fixed)
	cout << "Generating texture. Radius = " << radius << endl;
	typedef unsigned char BYTE;
    BYTE*  tex  = new BYTE[imgSize * imgSize * 3];												// Local buffers to store the GL textures
    float* mask = new float[imgSize * imgSize * 2];
	glEnable(GL_TEXTURE_2D);

	//0. Create 'tex_mask' and 'tex_dt':
	float dt_max = cloud->DT_max;
	Color color;
    for (int i = 0; i < imgSize*imgSize; ++i)													// Generate visualization texture
		{
 			float dt  = cloud->siteDT[i];
			float rdt = dt;
			if (rdt>radius) rdt=radius;
			rdt = 1-pow(rdt/radius,1.0f);

            mask[2*i]   = 1;
            mask[2*i+1] = rdt;

			float r,g,b;
			float2rgb(dt/dt_max,r,g,b,color_mode);
            tex[i * 3 + 0] = r*255;
            tex[i * 3 + 1] = g*255;
            tex[i * 3 + 2] = b*255;
		}

	glBindTexture(GL_TEXTURE_2D, tex_mask);														//REMARK: We need to pass this as luminance-alpha. If passing only as alpha, GL would
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,imgSize,imgSize,0,GL_LUMINANCE_ALPHA,GL_FLOAT,mask);	//set the luminance to zero, by default (which next prevents using GL_MODULATE)

	glBindTexture(GL_TEXTURE_2D, tex_dt);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imgSize, imgSize, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
	int cnnnt = 0;
	//1. Create 'tex_color': Holds the interpolated image
	float norm = (false_positive_range)? false_positive_range : image->image_max;				//Normalize 'image' automatically or vs user-specified range
    for (int i = 0; i < imgSize; ++i){															//Generate visualization texture
        for (int j = 0; j < imgSize; ++j)
		{
			int   id   = j * imgSize + i;
			float val  = std::min(image->image->value(i,j)/norm,1.0f);
			float cert = image->certainty->value(i,j);
			float   dt = mask[2*id+1];
 
            //Cert is the relative distance to an triangle edge:
            //1 is is over an edge
            //0 is is too far away from it
            cert = cert * dt;

            color = colorMap.getColor(val);
          
            //Define luminance as the amount of white in this pixel:
            //Minimum cert -> maximum white and vice versa.
            float lum = 1;
            color.r = (1-cert)*lum + cert*color.r;
            color.g = (1-cert)*lum + cert*color.g;
            color.b = (1-cert)*lum + cert*color.b;	

            tex[id * 3 + 0] = color.r*255;
            tex[id * 3 + 1] = color.g*255;
            tex[id * 3 + 2] = color.b*255;
			////if(tex[id * 3 + 0]!=255&&tex[id * 3 + 1]!=255&&tex[id * 3 + 2]!=255)
			//// printf("%d,%d,%d\t",tex[id * 3 + 0],tex[id * 3 + 1],tex[id * 3 + 2]);
		}
	}
    //printf("cnnnt=%d\n",cnnnt);

	glBindTexture(GL_TEXTURE_2D, tex_color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imgSize, imgSize, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);

	glDisable(GL_TEXTURE_2D);
    delete[] tex;
	delete[] mask;
}



//drawColormap(tex);





void Display::mouseCb(int button, int state, int x, int y)
{
	int modif = glutGetModifiers();

    if (state == GLUT_UP)													//Mouse button release:
	switch (button)
    {
        case GLUT_LEFT_BUTTON:
		{
			if (interpolation_dir==0)										//Interpret the mouse-release as a (de)selection ONLY if we didn't move after mouse-press:
			{
				vector<int> nn;												//Find closest cloud-point to mouse
				vector<float> nd;
				float xf = (float(x))/scale - transX;
				float yf = (winSize-float(y))/scale - transY;
				Point2d pix(xf,yf);
				cloud->searchNN(pix,1,nn,&nd);								//!!!Must search in the current view, not in the Cartesian cloud per se.
				if (!(modif & GLUT_ACTIVE_SHIFT)) selection.clear();		//Clear selection, if not in additive mode

				if (nd[0]>MAX_SELECT_DIST*MAX_SELECT_DIST)					//Clicked too far from any point: no selection
				{
					selected_point_id = -1;
					selected_group.clear();
				}
				else														//Clicked close enough to a point: select something:
				{
					selected_point_id = nn[0];								//Remember closest point to mouse

					if (modif & GLUT_ACTIVE_CTRL)							//CTRL-click: select all points in group under mouse
					{
					  //!!Must do sth else if we select in the histogram, e.g. locate closest point in histo, then get its coords in Cartesian, then use these for below.
					  int gid = visual_groups.cushionAtPoint(pix);			//Select all points in the label-group under the mouse
					  if (gid!=-1)
					  {														//Add selected points to current-selection
						   Grouping::PointGroup sel = visual_groups.groupAtPoint(gid);
						   for(Grouping::PointGroup::const_iterator it=sel.begin();it!=sel.end();++it)
							  selection.insert(*it);
						   selected_group.insert(gid);
					  }
					}
					else													//Normal click: add closest point to mouse to selection
					  selection.insert(selected_point_id);
				}

				if (ui_selection_size)										//3. Update Statistics UI (if any)
				{
					char buf[128];
					sprintf(buf,"%d",(int)selection.size());
					ui_selection_size->set_text(buf);
				}

				cloud->computeFalseNegatives(selection,false_negative_range);	//4. Update all visualizations that depend on 'selection'
				computePointColors(show_particles);
				computeDistribution();
				computeLabelMixing();
				computeFalseNegativesGraph();
				computeBundles();
				glui->sync_live();											//Force updating the GL window
			}

            isLeftMouseActive = false;
			interpolation_dir = 0;
			if (interpolation_level>0.5)
			{
			    SimplePointCloud* tmp = current_view;
				current_view = next_view;
				next_view = tmp;
			}
			interpolation_level = 0;

		}
		break;

        case GLUT_RIGHT_BUTTON:
            isRightMouseActive = false;
            break;
    }

    if (state == GLUT_DOWN)												//Mouse button click:
    {
        oldMouseX = x;
        oldMouseY = y;

        switch (button)
        {
        case GLUT_LEFT_BUTTON:											//Left button click:
		{
            isLeftMouseActive = true;
            break;
		}
        case GLUT_RIGHT_BUTTON:											//Right button click:
            isRightMouseActive = true;
            break;
        }
    }

	glutPostRedisplay();
}





void Display::drawBrush()												//Interactive data brushing. Closest point in cloud is already stored in 'closest_point'
{
	if (closest_point==-1 || !show_brush) return;

	const Point2d& closest = cloud->points[closest_point];

	glPointSize(5);														//1. Draw closest point
	glColor3f(1,0,0);
	glBegin(GL_POINTS);
	glVertex2f(closest);
	glEnd();
	glPointSize(1);

	const float rad = cloud->avgdist;
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glColor4f(0,0,0,0.4);
	drawCircle(closest,rad);
	glDisable(GL_BLEND);

	int tri = cloud->hitTriangle(brush_point);
	if (tri>=0)
	{
		const Triangle& tr = cloud->triangles[tri];
		glColor3f(1,1,1);													//2. Draw Delaunay triangle containing mouse cursor
		glLineWidth(3);														//   (if any was found..)
		const Point2d& p0 = cloud->points[tr(0)];
		const Point2d& p1 = cloud->points[tr(1)];
		const Point2d& p2 = cloud->points[tr(2)];
		glBegin(GL_LINES);
		glVertex2f(p0);
		glVertex2f(p1);
		glVertex2f(p0);
		glVertex2f(p2);
		glVertex2f(p1);
		glVertex2f(p2);
		glEnd();
		glLineWidth(1);
	}

	const PointCloud::EdgeMatrix::Row& row = (*cloud->sorted_edges)(closest_point);
	glColor3f(0,0,0);
	for(PointCloud::EdgeMatrix::Row::const_iterator it = row.begin();it!=row.end();++it)
	{
		int   j = it->pid;
		const Point2d& pj = cloud->points[j];

		glBegin(GL_LINES);
		glVertex2f(closest);
		glVertex2f(pj);
		glEnd();

		glRasterPos2f(pj.x,pj.y);

		char buf[100];
		sprintf(buf,"%d",j);
		glutDrawString(buf);
	}
}



void Display::passivemotionCb(int x,int y)
{
	if (x<0 || y<0 || x>winSize-1 || y>winSize-1) return;	//Don't track mouse if outside window

	brush_point = Point2d(x,winSize-y);
	brush_point.x /= scale;									//Transform back from pixel coordinates to world coordinates
	brush_point.y /= scale;
	brush_point.x -= transX;
	brush_point.y -= transY;

	vector<int> res;										//Find cloud point closest to the mouse position
	cloud->searchNN(brush_point,1,res);
	closest_point = res[0];									//Remember that point for drawing the brush later

	glutPostRedisplay();
}



void Display::motionCb(int x,int y)
{
	if (x<0 || y<0 || x>winSize-1 || y>winSize-1) return;	//Don't track mouse if outside window

    if (isLeftMouseActive)									//Motion with left-button pressed:
	{
															//Change interpolation level:
		if (x!=oldMouseX)
		{
		   if (interpolation_dir==0)						//First motion after left-click: record the moving direction (left or right)
		      interpolation_dir = (x<oldMouseX)? -1 : 1;	//We will only interpolate when the motion goes at that side of oldMouseX

		   if ((x<oldMouseX && interpolation_dir==-1) || (x>oldMouseX && interpolation_dir==1))
		      interpolation_level = std::min(2*fabs(x-oldMouseX)/winSize,1.0);

           //!!transX += double(x - oldMouseX) / scale;
           //!!transY -= double(y - oldMouseY) / scale;
           glutPostRedisplay();
		}
    }
    else if (isRightMouseActive)
	{
        //!!!scale -= (y - oldMouseY) * scale / 400.0;
        //!!!glutPostRedisplay();
    }
}



void Display::computeLabelMixing()							//Compute smooth label-mixing image 'tex_mixing' from the per-point label-mixing metric stored in the cloud.
{
	float* out_image = new float[winSize*winSize*4];

	const float* point_data = &cloud->label_mix[0];				//1. Compute label-mixing image (->'tex_mixing')
	ImageInterpolator::shepard(*cloud,point_data,out_image,shepard_averaging,color_mode);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,tex_mixing);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,winSize,winSize,0,GL_RGBA,GL_FLOAT,out_image);
	glDisable(GL_TEXTURE_2D);

	point_data = &cloud->aggregate_error[0];					//2. Compute aggregate (FP+FN) error image (->'agregate_error')
	ImageInterpolator::shepard(*cloud,point_data,out_image,shepard_averaging,color_mode);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,tex_aggregate);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,winSize,winSize,0,GL_RGBA,GL_FLOAT,out_image);
	glDisable(GL_TEXTURE_2D);


	point_data = &cloud->false_negative_error[0];				//3. Compute FN error image (->'tex_false_negatives')
	ImageInterpolator::shepard(*cloud,point_data,out_image,shepard_averaging,color_mode);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,tex_false_negatives);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,winSize,winSize,0,GL_RGBA,GL_FLOAT,out_image);
	glDisable(GL_TEXTURE_2D);

	delete[] out_image;
}




void Display::controlCb(int ctrl)
{
	switch(ctrl)
	{
	case UI_QUIT:
		exit(0);
		break;
	case UI_SCALE_DOWN:
		scale *= 0.9;
		break;
	case UI_SCALE_UP:
		scale *= 1.1;
		break;
	case UI_COLOR_MODE:
		generateTexture();
		break;
    case UI_CUSHION_STYLE:
		visual_groups.cushion_type = VisualGrouping::CUSHION_TYPE(cushion_style);
	    visual_groups.makeTextures();
		break;
	case UI_CUSHION_THICKNESS:
	    visual_groups.cushion_shading_thickness = cushion_shading_thickness;
		visual_groups.makeTextures();
		break;
	case UI_FALSEPOS_DISTWEIGHT:
	case UI_FALSEPOS_RANGE:
		generateTexture();
		break;
	case UI_FALSENEG_RANGE:
		cloud->computeFalseNegatives(selection,false_negative_range);
		computeLabelMixing();
		break;
	case UI_AGGREGATE_ERR_RANGE:
		cloud->computeAggregateError(aggregate_error_range);
		computeLabelMixing();
		break;
	case UI_SHEPARD_AVERAGING:
		computeLabelMixing();
		break;
	case UI_POINT_RADIUS:
		generateTexture();
		computeAllCushions();
		cloud->computeLabelMixing();
		computeLabelMixing();
		break;
	case UI_SHOW_POINTS:
		computePointColors(show_particles);
		computeDistribution();
		break;
	case UI_SCATTER_X_AXIS:
	case UI_SCATTER_Y_AXIS:
		computeDistribution();
		break;
	case UI_CUSHION_OPENING:
		computeAllCushions();
		break;
	case UI_GROUP_FINER:								//Refine cloud: if refinement exists, use it.
	{													//If not, we're already at the finest level, so nowhere to refine
		if (visual_clustering->finer)
	    {
		  visual_clustering = visual_clustering->finer;
		  computeGroupMeshes(visual_clustering);
	    }
	    break;
	}
	case UI_GROUP_COARSER:								//Coarsen cloud: use existing coarser-cloud, or coarsen on demand
    {
		if (!visual_clustering->coarser)
		   visual_clustering = visual_clustering->coarsen();
		else visual_clustering = visual_clustering->coarser;
	    computeGroupMeshes(visual_clustering);
		break;
    }
	case UI_RECOMPUTE_CUSHIONS:
		computeAllCushions();
		break;
	case UI_BUNDLE_ITERATIONS:
	case UI_BUNDLE_KERNEL:
	case UI_BUNDLE_EDGERES:
	case UI_BUNDLE_SMOOTH:
	case UI_BUNDLE_CPU_GPU:
		computeBundles();
		break;
	case UI_BUNDLE_DENS_ESTIM:
		bund->density_estimation = (CPUBundling::DENSITY_ESTIM)density_estimation;
		computeBundles();
		break;
	case UI_BUNDLE_FALSE_POS:
		computeFalseNegativesGraph();
		computeBundles();
		break;
	case UI_BUNDLE_SAVE:
		false_negatives_bundling->saveTrails("false_negatives.trl",false);
		break;
	}


	glui->post_update_main_gfx();						//Update GLUT window upon any parameter change (i.e., redraw)
}


void Display::keyboardCb(unsigned char k,int,int)
{
  switch (k)
  {
	case 'q':  controlCb(UI_QUIT);				break;
    case '.':  controlCb(UI_SCALE_DOWN);		break;
	case ',':  controlCb(UI_SCALE_UP);			break;
	case 't':  tex_interp = !tex_interp;		break;
	case 'c':  color_mode = !color_mode;
			   generateTexture();				break;
	case 'b':  show_brush = !show_brush;		break;
	case ' ':  show_maptype = (show_maptype+1)%7;
			   break;
	case '-': case '_':
			   controlCb(UI_GROUP_FINER);		break;
	case '+': case '=':
			   controlCb(UI_GROUP_COARSER);		break;
	case 'g':  show_groups = (show_groups+1)%5; break;
  }

  glui->sync_live();
  glui->post_update_main_gfx();
}


void Display::keyboard_cb(unsigned char k,int x,int y)
{
	instance->keyboardCb(k,x,y);
}

void Display::control_cb(int ctrl)									//Static entry point for UI events; calls non-static method
{
	instance->controlCb(ctrl);
}

void Display::display_cb()											//Driver callback for GLUT
{
	instance->displayCb();
}

void Display::mouse_cb(int button, int state, int x, int y)
{
	instance->mouseCb(button,state,x,y);
}

void Display::motion_cb(int x,int y)
{
	instance->motionCb(x,y);
}

void Display::passivemotion_cb(int x,int y)
{
	instance->passivemotionCb(x,y);
}

 

PointCloud* Display::loadPointCloud(char filename[]) {

	
    PointCloud *newCloud = new PointCloud(imgSize);
 
	printf("aaaaaaaaaaaaaaaaaaaaaaaaaa");
    newCloud->myLoadPex(filename, pointfile, true);
 
    newCloud->myFitToSize(pointcloud_minx, pointcloud_miny, pointcloud_maxx, pointcloud_maxy);
 
	newCloud->initEnd();
    
    //RadialGrouping* rg = new RadialGrouping(newCloud);						
    //From now on, use only the cleaned-up points	
    //RadialGrouping* crs = rg->coarsen(0);								
    //PointCloud* clean_cloud = crs->cloud;
 
    newCloud->avgdist = point_influence_radius;
 
    //delete rg;
    //delete newCloud;
    
//    delete visual_clustering;
//    delete labelg;
    
//    visual_clustering = new RadialGrouping(clean_cloud);
//    labelg = clean_cloud->groupByLabel();    
 

    if (dimrank_reduce_dimensions != 0)
        newCloud->myReduceAttributes(dimrank_reduce_dimensions_value);
    
    return newCloud;
    
}
















