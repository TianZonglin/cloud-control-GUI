#include <GL/glew.h>

#include "include/cpubundling.h"
#include "include/gdrawing.h"
#include "include/glutwrapper.h"
#include "include/gluiwrapper.h"

#include <cuda_gl_interop.h>
#include <math.h>
#include <iostream>
#include <string>
#include <time.h>


using namespace std;



GraphDrawing*		gdrawing_orig;                                                      //Original drawing, as read from the input file. Never changed.
GraphDrawing*		gdrawing_bund;                                                      //Bundled drawing, computed from gdrawing_orig.
GraphDrawing*		gdrawing_final;                                                     //Final drawing, done by relaxation-interpolation between gdrawing_orig and gdrawing_bund.
CPUBundling*		bund;
static float		scale = 1;                                                          //Scaling of coords from input file to screen pixels
static float		transX = 0, transY = 0;                                             //Translation of bbox of coords from input file to screen window

static int			fboSize;                                                            //Size of window (pow of 2)
static float		relaxation = 0;                                                     //Interpolation between gdrawing_orig and gdrawing_bund (in [0,1])
static float        max_displacement = 0.2;                                             //Max displacement allowed for a bundled-edge sampling point w.r.t. gdrawing_orig
static int          displ_rel_edgelength = 1;                                           //If true, max_displacement is fraction of fboSize; else max_displacement is fraction of current edge-length
static float		shading_radius = 3;                                                 //Radius of kernel used to normalize density-map for shading computations (pixels)
static int			show_points	= 0;                                                    //Show edge sample points or not
static int			show_edges	= 1;                                                    //Show graph edges or not
static int			show_endpoints = 0;                                                 //Show edge endpoints or not
static int			gpu_bundling = 1;                                                   //1 = do GPU bundling; 0 = do CPU bundling
static int          auto_update = 1;                                                    //1 = rebundle upon any relevant parameter change; 0 = ask user to explicitly press 'Bundle 1x'
static int			density_estimation = 1;                                             //Type of GPU KDE density estimation: 0=exact (using atomic ops); 1=fast (no atomics)
static int			color_mode = GraphDrawing::RAINBOW;                                 //Color map used to map various data fields to color
static int			alpha_mode = GraphDrawing::ALPHA_CONSTANT;                          //How to map edge values to alpha
static int			polyline_style = 0;                                                 //Bundle using polyline style or not
static int			bundle_shape = 0;                                                   //Shape of bundles (0=FDEB, 1=HEB)
static int			tangent = false;                                                    //
static int			block_endpoints = 1;                                                //Do not allow edgepoints to move during bundling (typically what we want)
static int			use_density_alpha = 0;                                              //Modulate final drawing's alpha by edge-density map or not
static int			shading = 0;                                                        //Use shading of bundles or not
static int			shading_tube = 0;                                                   //Style of shading (0=tube, 1=Phong)
static float		dir_separation = 0;                                                 //Separate different-direction bundles in the 'tracks' style or not
static int			draw_background = 0;                                                //Draw background image under graph or not
static const char*  save_filename = "CUBu_bundling.trl";                                //Name of text-file where to save bundling result

static GLUI*		glui;

enum 
{
		UI_SHOW_BUNDLES,
		UI_BUNDLE_ITERATIONS,
		UI_BUNDLE_MS_ITERATIONS,
		UI_BUNDLE_KERNEL,
		UI_BUNDLE_MS_KERNEL,
		UI_BUNDLE_EDGERES,
		UI_BUNDLE_SMOOTH,
		UI_BUNDLE_SMOOTH_ITER,
		UI_BUNDLE_SPEED,
		UI_BUNDLE_DENS_ESTIM,
		UI_BUNDLE_COLOR_MODE,
		UI_BUNDLE_ALPHA_MODE,
		UI_BUNDLE_CPU_GPU,
        UI_BUNDLE_AUTO_UPDATE,
		UI_BUNDLE_SHAPE,
		UI_BUNDLE_SHOW_POINTS,
		UI_BUNDLE_SHOW_EDGES,
		UI_BUNDLE_SHOW_ENDPOINTS,
		UI_BUNDLE_LINEWIDTH,
		UI_BUNDLE_GLOBAL_ALPHA,
		UI_BUNDLE_BLOCK_ENDS,
		UI_BUNDLE_POLYLINE_STYLE,
		UI_BUNDLE_SMOOTH_ENDS,
		UI_BUNDLE_RELAXATION,
        UI_BUNDLE_CLAMP,
        UI_BUNDLE_CLAMP_ABS_REL,
		UI_BUNDLE_DIRECTIONAL,
		UI_BUNDLE_DIR_REPULSION,
		UI_BUNDLE_DENSITY_ALPHA,
		UI_BUNDLE_DIR_SEPARATION,
		UI_BUNDLE_USE_SHADING,
		UI_BUNDLE_SHADING_RADIUS,
		UI_BUNDLE_SHADING_TUBE,
		UI_BUNDLE_SHADING_AMBIENT,
		UI_BUNDLE_SHADING_DIFFUSE,
		UI_BUNDLE_SHADING_SPECULAR,
		UI_BUNDLE_SHADING_SPECULAR_SIZE,
		UI_BUNDLE_BACK_IMAGE,
        UI_SAMPLE,
        UI_BUNDLE,
        UI_COPY_DRAWING_TO_ORIG,
        UI_RESET,
        UI_SAVE,
		UI_QUIT
};


void display_cb();
void buildGUI(int);
void bundle(int force_update=0);
void postprocess();



int main(int argc,char **argv)
{
	char* graphfile = 0;
	fboSize   = 512;	
	bool only_endpoints = false;
	int  max_edges = 0;

	for (int ar=1;ar<argc;++ar)
	{
		string opt = argv[ar];
		if (opt=="-f")											//Input file name:
		{
			++ar;
			graphfile = argv[ar];
		}
		else if (opt=="-i")										//Output image size:
		{
			++ar;
			fboSize = atoi(argv[ar]);
		}
		else if (opt=="-e")										//Use only trail endpoints:
		{
			only_endpoints = true;
		}	
		else if (opt=="-n")										//Use only max so-many-edges from input:
		{
			++ar;
			max_edges = atoi(argv[ar]);
		}
		
	}
	
    glutInitWindowSize(fboSize, fboSize); 
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA); 
    glutInit(&argc, argv); 
	int mainWin = glutCreateWindow("Graph bundling"); 
	
    glewInit();                                                 //Must initialize GLEW if we want to use CUDA-OpenGL interop, apparently
    cudaGLSetGLDevice(0);

    
	gdrawing_bund  = new GraphDrawing();						//Make two graphs: the original one, and the bundled one
	gdrawing_orig  = new GraphDrawing();				
	gdrawing_final = new GraphDrawing();

	bool ok = gdrawing_orig->readTrails(graphfile,only_endpoints,max_edges);	//Read some input graph
	if (!ok) 
	{
		cout<<"Error: cannot open file "<<graphfile<<endl;
		exit(1);
	}
	
	bund = new CPUBundling(fboSize);							//Create bundling engine; we'll use it for several graph bundling tasks in this class
	bund->block_endpoints = block_endpoints;
	bund->polyline_style = polyline_style;
	bund->tangent = tangent;
	bund->density_estimation = (CPUBundling::DENSITY_ESTIM)density_estimation;

	gdrawing_orig->normalize(Point2d(fboSize,fboSize),0.1);		//Fit graph nicely in the graphics window
	gdrawing_orig->draw_points	   = show_points;
	gdrawing_orig->draw_edges      = show_edges;
	gdrawing_orig->draw_endpoints  = show_endpoints;
	gdrawing_orig->color_mode	   = (GraphDrawing::COLOR_MODE)color_mode;
	gdrawing_orig->alpha_mode	   = (GraphDrawing::ALPHA_MODE)alpha_mode;
	gdrawing_orig->densityMap      = bund->h_densityMap;
	gdrawing_orig->shadingMap      = bund->h_shadingMap;
	gdrawing_orig->densityMapSize  = fboSize;
	gdrawing_orig->densityMax	   = &bund->densityMax;
	gdrawing_orig->use_density_alpha = use_density_alpha;
	gdrawing_orig->shading		   = shading;

    *gdrawing_bund  = *gdrawing_orig;                           //Init the bundled graph and the final (drawing) graph with the original graph.

    *gdrawing_final = *gdrawing_bund;                           //This ensures that all drawing options will work, even if we don't bundle anything next.

    
    glutDisplayFunc(display_cb);

	buildGUI(mainWin);

	glutMainLoop(); 	

	delete gdrawing_bund;
	delete gdrawing_orig;
	delete gdrawing_final;
    return 0;
}





void display_cb() 
{
    glClearColor(1,1,1,1);											//Reset main GL state to defaults
    glClear(GL_COLOR_BUFFER_BIT); 
    glDisable(GL_LIGHTING);											
    glDisable(GL_DEPTH_TEST); 

    glViewport(0,0,fboSize,fboSize); 
    glMatrixMode(GL_PROJECTION);									//Setup projection matrix
    glLoadIdentity(); 
    gluOrtho2D(0,fboSize,0,fboSize); 
    glMatrixMode(GL_MODELVIEW);										//Setup modelview matrix
    glLoadIdentity(); 
    glScalef(scale,scale,1);									
    glTranslatef(transX,transY,0);
	
	gdrawing_final->draw();											//Draw the final graph
	
    glutSwapBuffers();												//All done
}


void bundle(int force_update)                                       //Do a full new bundling of gdrawing_orig, save result into gdrawing_bund
{                                                                   //At the end of this, a bundled graph must be available in gdrawing_bund

    if (!force_update && !auto_update) return;                      //If auto_update is off, and we don't force bundling, nothing to do

    *gdrawing_bund = *gdrawing_orig;								//Copy the original drawing to gdrawing_bund, since we want to bundle it,
																	//but we don't want to alter the original drawing.
    
    bund->setInput(gdrawing_bund);									//Bundle the graph drawing

	if (gpu_bundling)												//Use the CPU or GPU method
		bund->bundleGPU();	
	else	
		bund->bundleCPU();
}

void postprocess()													//Postprocess the bundling before display
{
        cout<<"P start"<<endl;


	*gdrawing_final = *gdrawing_bund;								//Don't modify the bundled graph, copy it (because we want to redo postprocessing w/o redoing bundling)
	

	cout<<"Copy"<<endl;

	gdrawing_final->interpolate(*gdrawing_orig,relaxation,dir_separation,max_displacement,!displ_rel_edgelength);
																	//Relax bundling towards original graph, separate edge-directions,
                                                                    //and apply clamping of max-displacement (limit bundling)

        cout<<"Interp"<<endl;                                                                    
                                                                    

    bool needs_density = use_density_alpha || color_mode==GraphDrawing::DENSITY_MAP;																		
																		
	if (shading || needs_density)									//Compute density+optionally shading of relaxed graph
		bund->computeDensityShading(gdrawing_final,shading_radius,shading,shading_tube);		

	cout<<"shading, P end"<<endl;	

}



void control_cb(int ctrl)
{
	switch(ctrl)
	{
	case UI_BUNDLE_ITERATIONS:
	case UI_BUNDLE_MS_ITERATIONS:
	case UI_BUNDLE_KERNEL:
	case UI_BUNDLE_MS_KERNEL:
	case UI_BUNDLE_EDGERES:
	case UI_BUNDLE_SMOOTH:
	case UI_BUNDLE_SMOOTH_ITER:
	case UI_BUNDLE_SMOOTH_ENDS:
	case UI_BUNDLE_SPEED:
	case UI_BUNDLE_CPU_GPU:
	case UI_BUNDLE_DIR_REPULSION:
		bundle();
		break;
	case UI_BUNDLE_DENS_ESTIM:
		bund->density_estimation = (CPUBundling::DENSITY_ESTIM)density_estimation;
		bundle();
		break;	
	case UI_BUNDLE_SHAPE:
		bund->initEdgeProfile((CPUBundling::EDGE_PROFILE)bundle_shape);
		bundle();
		break;
	case UI_BUNDLE_BLOCK_ENDS:
		bund->block_endpoints = block_endpoints;
		bund->initEdgeProfile((CPUBundling::EDGE_PROFILE)bundle_shape);
		bundle();
		break;		
	case UI_BUNDLE_POLYLINE_STYLE:
		bund->polyline_style = polyline_style;
		bundle();
		break;		
	case UI_BUNDLE_DIRECTIONAL:
		bund->tangent = tangent;
		bundle();
		break;		
	case UI_BUNDLE_COLOR_MODE:
		gdrawing_orig->color_mode = gdrawing_bund->color_mode = (GraphDrawing::COLOR_MODE)color_mode;
		break;	
	case UI_BUNDLE_ALPHA_MODE:
		gdrawing_orig->alpha_mode = gdrawing_bund->alpha_mode = (GraphDrawing::ALPHA_MODE)alpha_mode;
		break;	
	case UI_BUNDLE_SHOW_POINTS:
		gdrawing_orig->draw_points = gdrawing_bund->draw_points = show_points;
		break;	
	case UI_BUNDLE_SHOW_EDGES:
		gdrawing_orig->draw_edges = gdrawing_bund->draw_edges = show_edges;
		break;
	case UI_BUNDLE_SHOW_ENDPOINTS:
		gdrawing_orig->draw_endpoints = gdrawing_bund->draw_endpoints = show_endpoints;
		break;	
	case UI_BUNDLE_DENSITY_ALPHA:
		gdrawing_orig->use_density_alpha = gdrawing_bund->use_density_alpha = use_density_alpha;
		break;	
	case UI_BUNDLE_USE_SHADING:
		gdrawing_orig->shading = gdrawing_bund->shading = shading;
		break;	
	case UI_BUNDLE_GLOBAL_ALPHA:
		gdrawing_bund->global_alpha = gdrawing_orig->global_alpha;
		break;
	case UI_BUNDLE_LINEWIDTH:
		gdrawing_bund->line_width = gdrawing_orig->line_width;
		break;
	case UI_BUNDLE_SHADING_AMBIENT:
		gdrawing_bund->amb_factor = gdrawing_orig->amb_factor;	
		break;
	case UI_BUNDLE_SHADING_DIFFUSE:
		gdrawing_bund->diff_factor = gdrawing_orig->diff_factor;	
		break;
	case UI_BUNDLE_SHADING_SPECULAR:
		gdrawing_bund->spec_factor = gdrawing_orig->spec_factor;	
		break;
	case UI_BUNDLE_SHADING_SPECULAR_SIZE:
		gdrawing_bund->spec_highlight_size = gdrawing_orig->spec_highlight_size;	
		break;
	case UI_BUNDLE_BACK_IMAGE:
		gdrawing_bund->draw_background = gdrawing_orig->draw_background = draw_background;
		break; 	
    case UI_BUNDLE_AUTO_UPDATE:
        if (auto_update) bundle();
        break;

            
	case UI_BUNDLE_DIR_SEPARATION:
	case UI_BUNDLE_SHADING_RADIUS:
	case UI_BUNDLE_SHADING_TUBE:
	case UI_BUNDLE_RELAXATION:
    case UI_BUNDLE_CLAMP:
    case UI_BUNDLE_CLAMP_ABS_REL:
        break;
    case UI_SAMPLE:
        gdrawing_bund->resample(bund->spl);
        break;
    case UI_BUNDLE:
        bundle(1);
        break;
    case UI_RESET:
        *gdrawing_bund = *gdrawing_orig;
        break;
        case UI_COPY_DRAWING_TO_ORIG:
        *gdrawing_orig = *gdrawing_final;
        *gdrawing_bund = *gdrawing_final;
        break;
    case UI_SAVE:
        gdrawing_final->saveTrails(save_filename,true);
        cout<<"Bundled data saved to file: "<<save_filename<<endl;
        break;
	case UI_QUIT:
		exit(0);
		break;	
	}
	
	postprocess();															//Postprocess the bundling before display	

	glui->post_update_main_gfx();											//Post a redisplay
}


void buildGUI(int mainWin)
{
    GLUI_Control *o1, *o2, *o3, *o4, *o5, *o6, *o7, *o8, *o9, *o10, *o11;
    GLUI_Panel *pan,*pan2,*pan3;											//Construct GUI:
	GLUI_Scrollbar* scr;
	glui = GLUI_Master.create_glui("CUDA Bundling");		

	GLUI_Rollout* ui_bundling = glui->add_rollout("Bundling",false);		//4. Panel "Bundling":

	pan3 = glui->add_panel_to_panel(ui_bundling,"Main bundling");
	pan = glui->add_panel_to_panel(pan3,"",GLUI_PANEL_NONE);
	o1 = new GLUI_StaticText(pan,"Iterations");
    o2 = new GLUI_StaticText(pan,"Kernel size");
    o3 = new GLUI_StaticText(pan,"Smoothing factor");
    o4 = new GLUI_StaticText(pan,"Smoothing iterations");
    glui->add_column_to_panel(pan,false);

    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->niter,UI_BUNDLE_ITERATIONS,control_cb);
	scr->set_int_limits(0,40);
    o1->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->h,UI_BUNDLE_KERNEL,control_cb);
	scr->set_float_limits(3,40);
    o2->set_h(scr->h);
    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->lambda,UI_BUNDLE_SMOOTH,control_cb);
	scr->set_float_limits(0,1);
    o3->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->liter,UI_BUNDLE_SMOOTH_ITER,control_cb);
	scr->set_int_limits(0,10);
    o4->set_h(scr->h);

	pan3 = glui->add_panel_to_panel(ui_bundling,"Ends bundling");

    pan = glui->add_panel_to_panel(pan3,"",GLUI_PANEL_NONE);
    o1 = new GLUI_StaticText(pan,"Block endpoints");
    o2 = new GLUI_StaticText(pan,"Iterations");
    o3 = new GLUI_StaticText(pan,"Kernel size");
    o4 = new GLUI_StaticText(pan,"Smoothing factor");
    glui->add_column_to_panel(pan,false);

    new GLUI_Checkbox(pan,"",&block_endpoints,UI_BUNDLE_BLOCK_ENDS,control_cb);
    //o1->set_h(scr->h);
    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->niter_ms,UI_BUNDLE_MS_ITERATIONS,control_cb);
	scr->set_int_limits(0,40);
    o2->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->h_ms,UI_BUNDLE_MS_KERNEL,control_cb);
	scr->set_float_limits(3,80);
    o3->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->lambda_ends,UI_BUNDLE_SMOOTH_ENDS,control_cb);
	scr->set_float_limits(0,1);	
    o4->set_h(scr->h);

	pan2 = glui->add_panel_to_panel(ui_bundling,"General options");

	pan = glui->add_panel_to_panel(pan2,"",GLUI_PANEL_NONE);
	o1  = new GLUI_StaticText(pan,"Edge sampling");
    o2  = new GLUI_StaticText(pan,"Advection speed");
    o3  = new GLUI_StaticText(pan,"Dir. bunding repulsion");
    o4  = new GLUI_StaticText(pan,"Relaxation");
    o5  = new GLUI_StaticText(pan,"Max bundle");
    o6  = new GLUI_StaticText(pan,"Max rel. to edgelength");
    o7  = new GLUI_StaticText(pan,"Polyline style");
    o8  = new GLUI_StaticText(pan,"Directional bundling");
    o9  = new GLUI_StaticText(pan,"GPU method");
    o10 = new GLUI_StaticText(pan,"Auto update");
    
    glui->add_column_to_panel(pan,false);

    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->spl,UI_BUNDLE_EDGERES,control_cb);
	scr->set_float_limits(3,50);
    o1->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->eps,UI_BUNDLE_SPEED,control_cb);
	scr->set_float_limits(0,1);
    o2->set_h(scr->h);
    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->rep_strength,UI_BUNDLE_DIR_REPULSION,control_cb);
    scr->set_float_limits(0,1);
    o3->set_h(scr->h);
    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&relaxation,UI_BUNDLE_RELAXATION,control_cb);
    scr->set_float_limits(0,1);
    o4->set_h(scr->h);
    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&max_displacement,UI_BUNDLE_CLAMP,control_cb);
    scr->set_float_limits(1.0f/1024,1);
    o5->set_h(scr->h);
    new GLUI_Checkbox(pan,"",&displ_rel_edgelength,UI_BUNDLE_CLAMP_ABS_REL,control_cb);
    //..o6->set_h(checkbox->h)
    new GLUI_Checkbox(pan,"",&polyline_style,UI_BUNDLE_POLYLINE_STYLE,control_cb);
    //..o7->set_h(checkbox->h)
    new GLUI_Checkbox(pan,"",&tangent,UI_BUNDLE_DIRECTIONAL,control_cb);
    //..o8->set_h(checkbox->h)
    new GLUI_Checkbox(pan,"", &gpu_bundling,UI_BUNDLE_CPU_GPU,control_cb);
    new GLUI_Checkbox(pan,"", &auto_update,UI_BUNDLE_AUTO_UPDATE,control_cb);
    
    
    pan3 = glui->add_panel_to_panel(pan2,"",GLUI_PANEL_NONE);
        pan = glui->add_panel_to_panel(pan3,"Density estimation");
        glui->add_column_to_panel(pan3,false);
        GLUI_RadioGroup* ui_dens_estim = new GLUI_RadioGroup(pan,&density_estimation,UI_BUNDLE_DENS_ESTIM,control_cb);
        new GLUI_RadioButton(ui_dens_estim,"Exact");
        new GLUI_RadioButton(ui_dens_estim,"Fast");

        pan = glui->add_panel_to_panel(pan3,"Bundle shape");
        glui->add_column_to_panel(pan3,false);
        GLUI_RadioGroup* ui_bundle_shape = new GLUI_RadioGroup(pan,&bundle_shape,UI_BUNDLE_SHAPE,control_cb);
        new GLUI_RadioButton(ui_bundle_shape,"FDEB");
        new GLUI_RadioButton(ui_bundle_shape,"HEB");
    //pan3 ready ----------
    
    
    pan = glui->add_panel("",GLUI_PANEL_NONE);                              //5. Buttons bar:
    new GLUI_Button(pan,"Resample",UI_SAMPLE,control_cb);
    glui->add_column_to_panel(pan,false);
    new GLUI_Button(pan,"Bundle",UI_BUNDLE,control_cb);
    glui->add_column_to_panel(pan,false);
    new GLUI_Button(pan,"Reset",UI_RESET,control_cb);
    glui->add_column_to_panel(pan,false);
    new GLUI_Button(pan,"Quit",UI_QUIT,control_cb);
    glui->add_column_to_panel(pan,false);
    pan = glui->add_panel("",GLUI_PANEL_NONE);                              //5. Buttons bar:
    new GLUI_Button(pan,"Result->input",UI_COPY_DRAWING_TO_ORIG,control_cb);
    glui->add_column_to_panel(pan,false);
    new GLUI_Button(pan,"Save",UI_SAVE,control_cb);
    glui->add_column_to_panel(pan,false);

    
	glui->add_column(true);													//--------------------------------------------------------
	GLUI_Rollout* ui_drawing = glui->add_rollout("Drawing",false);			//6. Roll-out "Drawing":
	pan = glui->add_panel_to_panel(ui_drawing,"Draw what");
	new GLUI_Checkbox(pan,"Edges", &show_edges,UI_BUNDLE_SHOW_EDGES,control_cb);			
	new GLUI_Checkbox(pan,"Control points", &show_points,UI_BUNDLE_SHOW_POINTS,control_cb);			
	new GLUI_Checkbox(pan,"End points", &show_endpoints,UI_BUNDLE_SHOW_ENDPOINTS,control_cb);			
    new GLUI_Checkbox(pan,"Background image",&draw_background,UI_BUNDLE_BACK_IMAGE,control_cb);
    
	pan = glui->add_panel_to_panel(ui_drawing,"",GLUI_PANEL_NONE);
	o1  = new GLUI_StaticText(pan,"Line width");
    o2  = new GLUI_StaticText(pan,"Global alpha");
    o3  = new GLUI_StaticText(pan,"Shading smoothing");
    o4  = new GLUI_StaticText(pan,"Shading ambient");
    o5  = new GLUI_StaticText(pan,"Shading diffuse");
    o6  = new GLUI_StaticText(pan,"Shading specular");
    o7  = new GLUI_StaticText(pan,"Shading highlight");
    o8  = new GLUI_StaticText(pan,"Density-modulated alpha");
    o9  = new GLUI_StaticText(pan,"Illumination");
    o10 = new GLUI_StaticText(pan,"Tube-style shading");
    o11 = new GLUI_StaticText(pan,"Direction separation");

    glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->line_width,UI_BUNDLE_LINEWIDTH,control_cb);
	scr->set_float_limits(1,5);
    o1->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->global_alpha,UI_BUNDLE_GLOBAL_ALPHA,control_cb);
	scr->set_float_limits(0,1);
    o2->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&shading_radius,UI_BUNDLE_SHADING_RADIUS,control_cb);
	scr->set_float_limits(2,15);
    o3->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->amb_factor,UI_BUNDLE_SHADING_AMBIENT,control_cb);
	scr->set_float_limits(0,1);
    o4->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->diff_factor,UI_BUNDLE_SHADING_DIFFUSE,control_cb);
	scr->set_float_limits(0,1);
    o5->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->spec_factor,UI_BUNDLE_SHADING_SPECULAR,control_cb);
	scr->set_float_limits(0,20);
    o6->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->spec_highlight_size,UI_BUNDLE_SHADING_SPECULAR_SIZE,control_cb);
	scr->set_float_limits(1,20);
    o7->set_h(scr->h);

	new GLUI_Checkbox(pan,"",&use_density_alpha,UI_BUNDLE_DENSITY_ALPHA,control_cb);
	new GLUI_Checkbox(pan,"",&shading,UI_BUNDLE_USE_SHADING,control_cb);
	new GLUI_Checkbox(pan,"",&shading_tube,UI_BUNDLE_SHADING_TUBE,control_cb);
		
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&dir_separation,UI_BUNDLE_DIR_SEPARATION,control_cb);
	scr->set_float_limits(-20,20);
    o11->set_h(scr->h);
    
    

    pan3 = glui->add_panel_to_panel(ui_drawing,"",GLUI_PANEL_NONE);
    
        pan = glui->add_panel_to_panel(pan3,"Coloring");
        glui->add_column_to_panel(pan3,false);
        GLUI_RadioGroup* ui_color = new GLUI_RadioGroup(pan,&color_mode,UI_BUNDLE_COLOR_MODE,control_cb);
        new GLUI_RadioButton(ui_color,"Grayscale (length)");
        new GLUI_RadioButton(ui_color,"Blue-red (length)");
        new GLUI_RadioButton(ui_color,"Red-blue (length)");
        new GLUI_RadioButton(ui_color,"Directional");
        new GLUI_RadioButton(ui_color,"Black");
        new GLUI_RadioButton(ui_color,"Density map");
        new GLUI_RadioButton(ui_color,"Displacement");

        pan = glui->add_panel_to_panel(pan3,"Transparency");
        glui->add_column_to_panel(pan3,false);
        GLUI_RadioGroup* ui_alpha = new GLUI_RadioGroup(pan,&alpha_mode,UI_BUNDLE_ALPHA_MODE,control_cb);
        new GLUI_RadioButton(ui_alpha,"Constant");
        new GLUI_RadioButton(ui_alpha,"Mark short edges");
        new GLUI_RadioButton(ui_alpha,"Mark long edges");


	glui->set_main_gfx_window(mainWin);										//Link GLUI with GLUT (seems needed)	
}
