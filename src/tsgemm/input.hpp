#pragma once

#include <hpx/program_options.hpp>

namespace tsgemm {

// Input
// note: has to be in main so that hpx knows about the various options
hpx::program_options::options_description init_desc();

} // end namespace tsgemm
