#ifndef _convection_h
#define _convection_h

#include "../atmosphere/atmosphere.h"


namespace ngam {


class Convection {
  public:
    virtual ~Convection() {}
    virtual void adjust(Atmosphere& atmosphere) = 0;
};


}

#endif
