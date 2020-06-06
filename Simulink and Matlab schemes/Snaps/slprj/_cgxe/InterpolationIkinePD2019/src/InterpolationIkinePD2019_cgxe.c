/* Include files */

#include "InterpolationIkinePD2019_cgxe.h"
#include "m_IeSKkqIJwqCfAF7PBwcYjD.h"

unsigned int cgxe_InterpolationIkinePD2019_method_dispatcher(SimStruct* S, int_T
  method, void* data)
{
  if (ssGetChecksum0(S) == 2995803437 &&
      ssGetChecksum1(S) == 4085775608 &&
      ssGetChecksum2(S) == 743299477 &&
      ssGetChecksum3(S) == 1395202479) {
    method_dispatcher_IeSKkqIJwqCfAF7PBwcYjD(S, method, data);
    return 1;
  }

  return 0;
}
