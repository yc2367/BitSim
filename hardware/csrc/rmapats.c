// file = 0; split type = patterns; threshold = 100000; total count = 0.
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "rmapats.h"

scalar dummyScalar;
scalar fScalarIsForced=0;
scalar fScalarIsReleased=0;
scalar fScalarIsDeposited=0;
scalar fNettypeIsForced=0;
scalar fNettypeIsReleased=0;
void  schedNewEvent (struct dummyq_struct * I1401, EBLK  * I1396, U  I622);
#ifdef __cplusplus
extern "C" {
#endif
void SinitHsimPats(void);
#ifdef __cplusplus
}
#endif