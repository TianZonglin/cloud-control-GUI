#include "include/config.h"

//general image settings
int canvasSize = 1024;


// input (if nothing is provided on command-line)
char* pointfile = "../data/segmentation";
char* projname = "lamp";
int selectedScalar = 0;
int timeframes_total = 0;

float pointcloud_minx = 1.0e+8;
float pointcloud_miny = 1.0e+8;
float pointcloud_maxx = -1.0e+8;
float pointcloud_maxy = -1.0e+8;

// lower-left (x,y), width and height of colormap bar
int scale_x = 10;
int scale_y = 10;
int scale_w = 150;
int scale_h = 10;

int histogramArea = 400;

bool floatStats = false;

// UI settings
int show_brush = 0;
int show_ranking_brush = 0;
int show_maptype = MAP_NOTHING;

//Dimension ranking
int   dimrank_metric = DIMRANK_CONTRIBUTION;
int   dimrank_strategy = DIMRANK_SIMILARITY;
int   dimrank_dissgroup1 = 1;
int   dimrank_dissgroup2 = 1;
float dimrank_radius = 0.1f;
float dimrank_radius_min = 0.01f;
int   dimrankUseBrightness = 1;
float dimRankBrightnessControl = 1;
int   dimrank_min_group_size = 2;                 //min # of elements to perform ranking in a neighborhood
float dimrank_contribution_threshold = 0.01;
float dimrank_filter_size = 0.7f;
int   draw_dimrank_heatmap = 0;
int   draw_dimrank_setmap = 0;
int   draw_dimrank_outlines = 0;
int   draw_outlines_wordcloud = 0;
float dimrank_outlines_diameter = 0.1f;
float dimrank_outlines_threshold = 0.7;
int   dimrankSetRegions = 0;
int   dimrank_reduce_dimensions = 0;
int   dimrank_reduce_dimensions_value = 10;

//Outlines
float outline_padding = 0.040f;
float outline_sampling_distance = 0.020;
int   outline_num_iterations = 10;
float outline_lambda = 0.5;


// Cloud drawing
int     show_particles = 0;
float   points_alpha   = 0.5;
float   cloud_color[]  = {0.0f, 0.0f, 1.0f};

// General settings
float   shepard_averaging = 5;  // alpha
float   point_influence_radius = 20;  // beta

// Colormap
int invert_colormap = 1;
int current_cmap = 5;

// Selection
int selected_point_id = -1;
int selected_point_group = 0;
int point_group_creation_index = 0;

int false_positive_distweight = 10;
float false_positive_range = 0;
float false_negative_range = 0;
float aggregate_error_range = 0.42;


// Bundling
float bundling_global_alpha = 1.0f;
float bundling_line_width = 1;
int   bundling_time_direction = 1; //both
int   bundling_time_alpha = 0;
int   bundling_timewindow_center = 0;
float bundling_timewindow_sigma = 1.0f;
int   bundling_timewindow_size = 5;
int   bundling_draw_selected_only = 0;
int   bundling_direction_brightness = 0;
float bundling_direction_brightness_length = 0.3f;