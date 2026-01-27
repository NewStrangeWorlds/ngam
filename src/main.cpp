
#include <iostream>

#include "config/global_config.h"
#include "spectral_grid/spectral_grid.h"

#include "object/brown_dwarf.h"


int main(int argc, char *argv[])
{
  bear::GlobalConfig config("browndwarf", "/media/data/opacity_data/helios-k/", "const_resolution", 10.0);

  bear::SpectralGrid spectral_grid(&config, 0.3, 100);
  
  bear::BrownDwarf brown_dwarf(&config, &spectral_grid);

  std::cout << "Hello, World!" << std::endl;

  std::cout << spectral_grid.nbSpectralPoints() << " spectral points in total.\n";
  
  return 0;
}
