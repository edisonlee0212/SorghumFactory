/*!**************************************************************************
\file    PDF6D.h
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The header file for the:  6D PDF indices database
******************************************************************************/

#ifndef PDF6D_h
#define PDF6D_h

//#################################################################################
//! \brief PDF6D - database of indices on 4D slices (planar x,y index + norm. params.)
//#################################################################################
class CPDF34D;
class CPDF4D;
//class CDataCache1D;

class CPDF6D {
private:
	int m_numOfRows;          //! no. of rows in spatial BTF index
	int m_numOfCols;          //! no. of columns in spatial BTF index
	int m_rowsOffset;       //! offset of the first row as we do not need to start from 0
	int m_colsOffset;       //! offset of the first column as we do not need to start from 0  
	int** m_pdf6DSlices;   //! planar index pointing on 4D PDF for individual pixels
	float** m_pdf6DScale; //! corresponding normalization values
	// the database of 4D functions to which we point in the array PDF6Dslices
	CPDF4D* m_pdf4;
	void Allocate(int nrows, int ncols);
	void CleanObject();
	// The shared coordinates to be used for interpolation
	// when retrieving the data from the database
	// This is required for SSIM data precomputation
	// the number of slices per phi (=3D functions) to represent one 4D function
	int m_slicesPerPhi;
	// the number of slices per theta (=2D functions) to represent one 3D function
	int m_slicesPerTheta;
	// the number of indices in parameter alpha
	int m_slicesPerHemisphere;
	// the number of values for 1D function
	int m_lengthOfSlice;
	// the number of colors
	int m_numOfColors;
public:
	CPDF6D(int nrows, int ncols, CPDF4D* PDF4);
	~CPDF6D();

	// sets ofset of the first pixel to be addressed correctly
	void SetOffset(int rows_offset, int cols_offset);

	// This is required for SSIM data precomputation - setting the dimensions
	void SetSizes(int nSlicesPerPhi, int nSlicesPerTheta, int nSlicesPerHemi,
		int nLengthOfSlice, int nncolour);

	// get a single value
	float Get(int y, int x, int indexPhi, int indexTheta, int posAlpha, int posBeta,
		int posLAB, TSharedCoordinates& tc) const;

	// get single value for arbitrary angles  and planar position
	void GetValDeg(int y, int x, float illuminationTheta, float illuminationPhi, float viewTheta, float viewPhi,
		float rgb[], TSharedCoordinates& tc) const;
	void GetValRad(int y, int x, float theta_i, float phi_i, float theta_v, float phi_v,
		float rgb[], TSharedCoordinates& tc) const;
	// Fast version, which is the same functionality, but less computations
	// of interpolation parameters
	void GetValDeg2(int y, int x, float illuminationTheta, float illuminationPhi, float viewTheta, float viewPhi,
		float rgb[], TSharedCoordinates& tc) const;
	void GetValRad2(int y, int x, float theta_i, float phi_i, float theta_v, float phi_v,
		float RGB[], TSharedCoordinates& tc) const;
	// Slow version with Shepard interpolation based on distances and weighting
	void GetValDegShepard(int y, int x, float theta_i, float phi_i, float theta_v, float phi_v,
		float RGB[], TSharedCoordinates& tc) const;

	// Shepard interpolation based on distances and weighting - another implementation
	void GetValDegShepard2(int y, int x, float theta_i, float phi_i, float theta_v, float phi_v,
		float RGB[], TSharedCoordinates& tc) const;

	// get an index of planar position x,y
	int GetIndex(int y, int x) const;
	// get scale of planar position x,y
	float GetScale(int y, int x) const;
	// get offset of planar position x,y
	void GetOffset(int& arows_offset, int& acols_offset) const {
		arows_offset = this->m_rowsOffset; acols_offset = this->m_colsOffset;
	}
	// gets the size of the two-dimensional array of BTF data
	void GetSize(int& pnrows, int& pncols) const {
		pnrows = this->m_numOfRows; pncols = this->m_numOfCols;
	}
	// returns the number of pixels by the material
	int GetCountPixels() const { return m_numOfRows * m_numOfCols; }
	//return memory in bytes required by the data representation
	int GetMemory() const;
	int GetMemoryQ() const;

	int Load(char* prefix, int MLF);

	//! \brief computes importance sampling given the coordinates and [0-1]^2
	// random variables. It compute the normalized direction.
	int ImportanceSamplingDeg(int irow, int jcol, float theta_v, float phi_v,
		float q0, float q1, float& theta_i, float& phi_i,
		TSharedCoordinates& tc);
	int ImportanceSamplingDeg(int iRow, int jCol, float viewTheta, float viewPhi,
		int cntRays, float q0Q1[], float illuminationThetaPhi[],
		TSharedCoordinates& tc) const;

	//! \brief computes albedo for fixed viewer direction
	void GetViewerAlbedoDeg(int irow, int jcol, float theta_v, float phi_v, float RGB[],
		TSharedCoordinates& tc);

};//--- CPDF6D ---------------------------------------------------------

#endif
