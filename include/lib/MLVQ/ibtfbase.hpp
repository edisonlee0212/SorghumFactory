/*!**************************************************************************
\file    ibtfbase.h
\author  Vlastimil Havran
\date    15/11/2010
\version 1.01

  The header file for the: BTFBASE project - interface

*****************************************************************************/

#ifndef __IBTFBASE_H__
#define __IBTFBASE_H__

// Given the name of the file or directory, it inits the data structures
// by reading them from file(s)
extern int InitBtfBase(const char* parName);

// It closes the data structures
extern void FinishBtfBase();

// If possible, it selects more materials compressed to the same structures
extern void SelectBtfMaterial(int materialIndex);

// It returns the index of currently selected material
extern int GetIndexOfSelectedBtfMaterial();

// It returns the number of BTF materials compressed to one database
extern int GetBtfMaterialCount();

// This returns the resolution of BTF images compressed to the current database
extern void GetSizeBtfBase(int& rowsNum, int& colsNum);

// It returns the name of the BTF data
extern const char* GetBtfMaterialName();

// It returns the value of BTF given spatial index, viewing direction and illumination
// direction specified by angles in degrees (theta: 0-90, phi: 0-360)
extern void GetBtfDeg(int iRow, int jCol, float illuminationTheta, float illuminationPhi,
	float viewTheta, float viewPhi, float rgb[]);

// It returns the value of BTF given spatial index, viewing direction and illumination
// direction specified by angles in radians (theta: 0-M_PI/2, phi: 0-2*M_PI)
extern void GetBtfRad(int iRow, int jCol, float illuminationTheta, float illuminationPhi,
	float viewTheta, float viewPhi, float rgb[]);

// For using more databases it sets currently valid database
extern void SetBtfBasePointer(void* newPointer);
// It returns the pointer to currently used database
extern void* GetBtfBasePointer();

// Make importance sampling - given spatial indices, viewing direction in degrees and
// two random numbers it generates illumination direction in degrees.
extern int ImportanceSamplingDegBtfBase(int iRow, int jCol, float viewTheta, float viewPhi,
	float q0, float q1, float& illuminationTheta, float& illuminationPhi);

// Make importance sampling - given spatial indices, viewing direction in degrees and
// a set of couples of random numbers it generates a set of illumination directions in degrees.
extern int ImportanceSamplingDegBtfBase(int iRow, int jCol, float viewTheta, float viewPhi,
	int cntRays, float q0Q1[], float illuminationThetaPhi[]);

// This computes the albedo given the viewing direction, it is the value between 0.0
// and 1.0 for each channel separately.
extern void GetViewerAlbedoDegBtfBase(int iRow, int jCol, float viewTheta, float viewPhi,
	float rgb[]);

// Given the angle indices in the new onion parametrization, it returns the
// light direction theta_i, phi_i and viewer direction theta_v, phi_v in
// degrees. It returns 0 on success, 1 if indices are wrong!
extern int GetParametrizationDegBtfBase(int indexBeta, int indexAlpha, int indexTheta,
	int indexPhi, float& illuminationTheta,
	float& illuminationPhi, float& viewTheta, float& viewPhi);

#endif //  __IBTFBASE_H__

