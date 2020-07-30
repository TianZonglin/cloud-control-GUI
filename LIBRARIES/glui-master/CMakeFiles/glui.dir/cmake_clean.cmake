file(REMOVE_RECURSE
  "libglui.pdb"
  "libglui.so.2.37"
  "libglui.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/glui.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
