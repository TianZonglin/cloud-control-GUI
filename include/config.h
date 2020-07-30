#ifndef CONFIG_H
#define	CONFIG_H

extern int canvasSize;

extern char*    pointfile;
extern char*    projname;
extern int      timeframes_total;
extern int      selectedScalar;
extern float    shepard_averaging;
extern int      scale_x;
extern int      scale_y;
extern int      scale_w;
extern int      scale_h;
extern int      histogramArea;
extern int      show_brush;
extern int      show_ranking_brush;
extern int      current_cmap;
extern float    point_influence_radius;
extern bool     floatStats;
extern float pointcloud_minx;
extern float pointcloud_miny;
extern float pointcloud_maxx;
extern float pointcloud_maxy;


enum MapType {MAP_FALSE_POSITIVES,
              MAP_FALSE_NEGATIVES,
              MAP_AGGREGATE_ERROR,
              MAP_LABEL_MIXING,
              MAP_DT,
              MAP_DIMENSION_RANKING,
              DYNAMIC_PROJECTION,
              MAP_NOTHING};

//Dimension ranking related 
enum RankingMetric {DIMRANK_CONTRIBUTION,
                    DIMRANK_VARIANCE,
                    DIMRANK_STORED};

enum RankingStrategy {DIMRANK_SIMILARITY,
                      DIMRANK_DISSIMILARITY};
                    
extern  float dimrank_radius;
extern  float dimrank_radius_min;
extern int    dimrank_metric;
extern int    dimrank_strategy;
extern int    dimrank_dissgroup1;
extern int    dimrank_dissgroup2;
extern int    dimrankUseBrightness;
extern float  dimRankBrightnessControl;
extern int    dimrank_min_group_size;
extern float  dimrank_contribution_threshold;
extern int    draw_dimrank_heatmap;
extern int    draw_dimrank_setmap;
extern float  dimrank_filter_size;
extern int    draw_dimrank_outlines;
extern float  dimrank_outlines_diameter;
extern float  dimrank_outlines_threshold;
extern int    draw_outlines_wordcloud;
extern int    dimrankSetRegions;
extern int   dimrank_reduce_dimensions;
extern int   dimrank_reduce_dimensions_value;


extern float outline_padding;
extern float outline_sampling_distance;
extern int   outline_num_iterations;
extern float outline_lambda;


extern int    show_maptype;
extern int    show_particles;
extern float  points_alpha;
extern int    invert_colormap;
extern float  cloud_color[3];
extern int    selected_point_id;
extern int    selected_point_group;
extern int    point_group_creation_index;
extern float  false_positive_range;
extern int    false_positive_distweight;
extern float  false_negative_range;
extern float  aggregate_error_range;

extern float bundling_global_alpha;
extern float bundling_line_width;
extern int   bundling_time_alpha;
extern  int   bundling_time_direction;
extern int   bundling_timewindow_center;
extern float bundling_timewindow_sigma;
extern int   bundling_timewindow_size;
extern int   bundling_direction_brightness;
extern float bundling_direction_brightness_length;

#endif