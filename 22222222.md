
BUILDING

========

The software was tested to build and run under Linux Ubuntu. As a prerequisite, CUDA 4.0 or higher should be installed.

To build: 

1. Go to LIBRARIES/glui-master. Possibly re-run CMake to update to your platform's configuration.
2. Do a "make" from the root directory

To clean everything:

1. Do a "make realclean" from the root directory.


RUNNING

=======

Example:

projwiz -f DATA/segmentation lamp 


DONE
====


* Better selection mechanism. We can now select points and groups-of-points. Selection is additive and can be reset. All selections work by clicking in the main window:
   -normal click: select closest point to mouse;
   -CTRL-click: select entire (label) group under the mouse;
   -SHIFT-click: add points to selection rather than overwriting it; works in both normal and CTRL modes;
   -click far away from any point: clear selection;

* False-negative bundles: Now they're done for either the entire dataset or the current selection:
   -the current selection is void: FN's are shown for the entire dataset;
   -the current selection is not void: only FN's of points in the selection are used;

* False-negative map: The map is now computed w.r.t. the current selection. That is:
   -if the crt-selection is 1 point: same result as the original idea (show FN-error for all other points w.r.t. selected point)
   -if the crt-selection is more points: for all points p not in selection, show _minimal_ FN-error w.r.t. _all_ points in the selection.


TODO
====

* Add possibility to select also groups from the visual clustering; for this, we must detect connected-components in the visual clustering.

* Add generic way to show min/max/avg of the various plotted signals (FNs, FPs, avg-error, etc)

* Show spatial-difference/error-difference between 2 projections
