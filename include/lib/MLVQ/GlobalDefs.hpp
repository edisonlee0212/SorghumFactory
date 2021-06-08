/*!**************************************************************************
\file    AuxFuncs.h
\author  Jiri Filip and Vlastimil Havran
\date    15/11/2006
\version 1.01

  The header file for the: BTFBASE project

*****************************************************************************/
#ifndef GlobalDefs_c
#define GlobalDefs_c

//#################################################################################
//! \brief Global variable definitions
//#################################################################################

#define FLOATMAX ((float)(1e15))
#define MAXINT ((int)(32700))
#define EPSILON 3.0e-7
#define PI 3.14159265358979323846

template <class T>
T Max(T a, T b) { return a > b ? a : b; }

template <class T>
T Min(T a, T b) { return a < b ? a : b; }

inline float
Square(const float a) { return a * a; }

#endif // GlobalDefs_c
