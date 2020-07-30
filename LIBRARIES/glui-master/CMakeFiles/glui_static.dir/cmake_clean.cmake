file(REMOVE_RECURSE
  "libglui_static.pdb"
  "libglui_static.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/glui_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
