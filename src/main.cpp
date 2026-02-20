
#include <iostream>

#include "config/global_config.h"
#include "spectral_grid/spectral_grid.h"

#include "object/brown_dwarf.h"


int main(int argc, char *argv[])
{
  ngam::GlobalConfig config("browndwarf", "/media/data/opacity_data/helios-k/", "const_resolution", 1000.0);
  
  ngam::SpectralGrid spectral_grid(&config, 0.3, 100);
  
  ngam::BrownDwarf brown_dwarf(&config, &spectral_grid);

  std::cout << "Hello, World!" << std::endl;

  brown_dwarf.computeAtmosphericStructure();
  
  return 0;
}
