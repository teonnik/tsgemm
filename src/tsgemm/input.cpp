#pragma once

#include <tsgemm/input.hpp>

namespace tsgemm {

hpx::program_options::options_description init_desc() {
  namespace po = hpx::program_options;
  po::options_description desc("Allowed options.");

  // clang-format off
  desc.add_options()
     ("len_m",      po::value<int>()->default_value( 100), "m dimension")
     ("len_n",      po::value<int>()->default_value( 100), "n dimension")
     ("len_k",      po::value<int>()->default_value(1000), "k dimension")
     ("tile_m",     po::value<int>()->default_value(  64), "tile m dimension")
     ("tile_n",     po::value<int>()->default_value(  64), "tile n dimension")
     ("tile_k",     po::value<int>()->default_value(  64), "tile k dimension")
     ("pgrid_rows", po::value<int>()->default_value(   1), "process grid rows")
     ("pgrid_cols", po::value<int>()->default_value(   1), "process grid columns")
     ("blk_rows",   po::value<int>()->default_value(  32), "block rows")
     ("blk_cols",   po::value<int>()->default_value(  32), "block columns")
  ;
  // clang-format on

  return desc;
}

} // end namespace tsgemm
