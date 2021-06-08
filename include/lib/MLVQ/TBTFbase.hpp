/*!**************************************************************************
\file    TBTFbase.h
\author  Jiri Filip
\date    13/11/2006
\version 1.01

  The header file for the:
	BTFBASE project with V. Havran (MPI)

*****************************************************************************/
#ifndef TBTFbase_c
#define TBTFbase_c

#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2
#define M_PI_4     0.785398163397448309616  // pi/4
#define M_1_PI     0.318309886183790671538  // 1/pi
#define M_2_PI     0.636619772367581343076  // 2/pi

#include <PDF1D.hpp>
#include <VectorColor.hpp>
#include <IndexAB.hpp>
#include <PDF2D.hpp>
#include <PDF3D.hpp>
#include <PDF4D.hpp>
#include <PDF6D.hpp>

//#################################################################################
//! \brief Class for BTF modelling 
//#################################################################################

struct TileMap
{
	int m_mWidth = 0;
	int m_mHeight;
	int* m_mOffsets = nullptr;
	void Clear() {
		delete m_mOffsets;
		m_mOffsets = nullptr;
	}
	bool Empty() const {
		return m_mOffsets == nullptr;
	}

	~TileMap();
};

class BTFbase
{
	bool m_allMaterialsInOneDatabase; //! if to compress all materials into one database
	//! if view direction represented directly by UBO measurement quantization
	bool m_use34ViewRepresentation;
	bool m_usePdf2CompactRep; //! If we do not separate colors and luminances for 2D functions
	int m_materialCount; //! how many materials are stored in the database
	int m_materialOrder; //! order of the material processed
	char* m_materialName; //! name of currently analysed BTF material
	int m_nRows, m_nCols; //! number of columns and rows in analysed original BTF image (not tile!)
	int m_rowsOffset, m_colsOffset; //! offset of the first pixel in the BTF image
	int m_numRows, m_numCols;     //! number of columns and rows to be analysed (<=nrows, <=ncols)
	int m_nColor;              //! number of spectral channels in BTF data
	char* m_bPath;        //! path to store the database with BTF data
	char* m_iPath;        //! path to input BTF data
	char* m_oPath;        //! path for storing some output images (debugging)
	char* m_tmpPath;      //! path for storing some temporary data
	//! array storing BTF UBO measurement angles and corresponding 3D points
	//  static float **angles;  
	float** m_btfTile; //! \brief One complete BTF tile [tileX*tileY*(ncolour+1)][nview*nillu]
	TSharedCoordinates m_tcTemplate;
	float m_stepAlpha; //! angular step between reflectance values in angle alpha (in degrees)
	bool m_useCosBeta; //! use cos angles

	int m_hdr;            // save image in: HRD (1) or PNG (0) -- current value
	int m_maxShift;       // 0-off, >0 maximal shift +-
	int m_metric;         // 0-Jeffrey,1-Weber

	int m_lengthOfSlice;  //! Number of measurement points on along slice parametrised by "beta"
	int m_slicesPerHemisphere;  //! Number of slices over hemisphere parametrized by "alpha"
	int m_noOfTheta;      //! number of different theta viewing angles stored in PDF3D
	int m_noOfPhi;        //! number of different phi viewing angles stored in PDF4D

	int m_maxPdf1D;       //! number of allocated 1D PDF L-slices
	int m_maxVectorColor; //! number of allocated CIE a-b colours
	int m_maxIndexSlices; //! number of allocated 1D colour index slices
	int m_maxPdf2D;       //! number of allocated 2D PDF indices
	int m_maxPdf2DLuminanceColor; //! number of allocated 2D PDF indices for color and luminance
	int m_maxPdf3D;       //! number of allocated 3D PDF indices
	int m_maxPdf4D;       //! number of allocated 4D PDF indices
	int m_maxPdf34D;       //! number of allocated 3-4D PDF indices for PDF34D
	bool m_minimumCacheStrategy; //! how the caches of PDF databases are used

	// database objects declarations
	CPDF1D* m_pdf1;     //! 1D PDF slices database
	CVectorColor* m_ab; //! CIE a-b colours database
	CIndexAB* m_iab;    //! 1D colour index slices database
	CPDF2D* m_pdf2;     //! 2D PDF indices database (illum dependent reflectance)
	CPDF3D* m_pdf3;     //! 3D PDF indices database (isotropic BRDF)
	CPDF4D* m_pdf4;     //! 4D PDF indices database (BRDF)
	CPDF34D* m_pdf34;     //! 4D PDF indices database (BRDF)
	CPDF6D* m_pdf6;     //! 6D PDF indices database (planar x,y) for current material

	// The tables of pointers to instances representing the materials
	CPDF1D** m_arPdf1;     //! 1D PDF slices database
	CVectorColor** m_arAb; //! CIE a-b colours database
	CIndexAB** m_arIab;    //! 1D colour index slices database
	CPDF2D** m_arPdf2;     //! 2D PDF indices database (illum dependent reflectance)
	CPDF3D** m_arPdf3;     //! 3D PDF indices database (isotropic BTF)
	CPDF4D** m_arPdf4;     //! 4D PDF indices database (BTF)
	CPDF34D** m_arPdf34;    //! 34D PDF indices database (BTF)
	CPDF6D** m_arPdf6;     //! 6D PDF indices database (planar x,y) for all materials
	char** m_arInputPath; //! path to input BTF data
	char** m_arOutputPath; //! path for storing some output images (debugging)
	char** m_arTempPath; //! path for storing some output images (debugging)
	char** m_arMaterialName; //! name of currently analysed BTF material
	bool* m_arHdr;           //! The information about HDR for the material
	float* m_arHdRvalue;     //! The multiplicator for HDR when computing RMS

	float m_mPostScale;
	TileMap m_tileMap; // tile map computed using BTF roller
	// ---------- FUNCTIONS -------------------------------------------------------

	//! \brief deletes individual databases except PDF6
	void DeleteDatabases(int matOrder);
	//! \brief allocates individual databases except PDF6
	void AllocateDatabases(int matOrder);
	//! \brief Deletes the paths for the indexed material
	void DeletePaths(int matOrder);
	//! \brief Allocate the paths for the indexed material
	void AllocatePaths(int matOrder, const char* nmaterialName,
		const char* niPath, const char* noPath,
		const char* ntmpPath);

	//! \brief sets some system variables such as hemisphere discretization
	void SetSystemVariables();
	//! \brief Allocates the tables for database instances for several materials;
	void AllocateArrays(int materialCount);
	void DeleteArrays();

	// Creates shared variables (tc) given the parameterization
	void CreateSharedVariables();
	// Sets the material status (HDR=true or LDR=false) 
	void SetHdrFlag(int materialIndex, bool flagHDR, float HDRvalue);
	// Save currently used settings to the specified file
	void SaveSettingsToLogfile(FILE* fp, char* p);

	// The size of header in items and bytes for the precomputed/saved BTF data
	// for one tile of size sizeTilePD x sizeTilePD, which is saved as a unit
	// to the disk in the directory given by tmpPath
	enum { sizeHeaderPDitems = 64, sizeHeaderPDbytes = 256 };

public:
	// constructor for only loading the database
	BTFbase(const char* basePath, bool isBRDFdata);
	~BTFbase();

	//! \brief returns the number of BTF materials in the database
	int GetMaterialCount() const { return m_materialCount; }

	//! \brief returns the order of the current material
	int GetMaterialOrder() const { return m_materialOrder; }

	//! \brief sets the order of the material to be used in the application functions
	// It cannot be used during compression !!!!
	int SelectMaterialForRendering(int order);
	const char* GetCurrentMaterialName() const { return m_materialName; }

	//! \brief sets the order of the material to be used in the functions
	void SetMaterial(int order);

	//! \brief sets information about material: name,path,no. of tiles and their sizes
	void SetMaterial(int matOrder, const char* nmaterialName, int HDRflag, float HDRvalue,
		const char* niPath, const char* noPath, const char* ntmpPath,
		int nrows, int ncols, int rows_offset, int cols_offset,
		int numRows, int numCols);
	//! \brief use cos angles
	bool UseCosBetaDiscretization() const { return m_useCosBeta; }

	// Sets the material status (HDR=true or LDR=false) 
	bool IsHdr() const { return m_hdr; }
	float GetHdRvalue() const { return m_arHdRvalue[m_materialOrder]; }

	//! \brief restores pixel BTF value for specified planar pos and view and illum directions
	// the angles are specified in DEGREEES, theta <0-90>, phi<0-360)
	void GetValDeg(int iRow, int jCol, float illuminationTheta, float illuminationPhi,
		float viewTheta, float viewPhi, float rgb[]) const;
	//! \brief restores pixel BTF value for specified planar pos and view and illum directions
	// the angles are given in radiances: theta <0-PI/2>, phi<0-2*PI)
	void GetValRad(int irow, int jcol, float theta_i, float phi_i,
		float theta_v, float phi_v, float RGB[]);
	//! \brief restores pixel BTF value for specified planar pos and view and illum directions
	// the angles are given in radiances: theta <0-PI/2>, phi<0-2*PI), the x and y are relative to
	// the resolution
	void GetValRadRel(const float y, const float x, float theta_i, float phi_i,
		float theta_v, float phi_v, float RGB[]);

	//! \brief restores pixel BTF value for specified planar pos and view and illum directions
	// the angles are specified in DEGREEES, theta <0-90>, phi<0-360) using Shepard interpolation
	// technique which provides smoother interpolant
	void GetValDegShepard(int irow, int jcol, float theta_i, float phi_i,
		float theta_v, float phi_v, float RGB[]);
	void GetValDegShepard2(int irow, int jcol, float theta_i, float phi_i,
		float theta_v, float phi_v, float RGB[]);

	// !\ brief implements importance sampling. It given random variables in range [0-1]x[0-1]
	// it generates a ray along this direction. It returns 0 on success, 1 on failure
	int ImportanceSamplingDeg(int irow, int jcol, float theta_v, float phi_v,
		float q0, float q1, float& theta_i, float& phi_i);

	int ImportanceSamplingRad(int irow, int jcol, float theta_v, float phi_v,
		float q0, float q1, float& theta_i, float& phi_i)
	{
		int result = ImportanceSamplingDeg(irow, jcol, theta_v * (180.0f / M_PI),
			phi_v * (180.0f / M_PI), q0, q1, theta_i, phi_i);
		theta_i *= M_PI / 180.0f;
		phi_i *= M_PI / 180.0f;
		return result;
	}

	int ImportanceSamplingDeg(int iRow, int jCol, float viewTheta, float viewPhi,
		int cntRays, float q0Q1[], float illuminationThetaPhi[]) const;

	//! \brief computes albedo for fixed viewer direction
	void GetViewerAlbedoDeg(int irow, int jcol, float theta_v, float phi_v, float RGB[]);

	void GetOffset(int& rows_offsetV, int& cols_offsetV) const
	{
		rows_offsetV = m_rowsOffset; cols_offsetV = m_colsOffset;
	}
	void GetSize(int& numRowsV, int& numColsV) const
	{
		numRowsV = m_numRows; numColsV = m_numCols;
	}

	//! \brief loads BTFBASE from files
	bool LoadBtfbase(const char* prefix, bool recover);

	void DeleteBtfs();

	//! \brief We compute the compresssion ratios for either all materials or only current
	// material specified by 'matOrderCurrent', where 'cntPixelsCurrent' were computed
	void ComputeSizes(
		// input
		bool onlyCurrentMaterial,
		int matOrderCurrent,
		int cntPixelsCurrent,
		// output
		long double& origSize,
		long double& compSize,
		long double& compSizeQuantized,
		// input ... compute quantized version - it can take time
		bool cq = false);

	// This given the indices to new onion parametrization computes the theta,phi angles
	// for light direction (theta_i, phi_i) and viewer direction (theta_v, phi_v)
	int GetParametrizationDegBtfBase(int indexBeta, int indexAlpha, int indexTheta, int indexPhi,
		float& theta_i, float& phi_i, float& theta_v, float& phi_v,
		TSharedCoordinates& tc);
};//--- BTFbase -----------------------------------------------------

// Only global variable within the project
//extern BTFbase *FW;
//extern int HDRflag;

#endif
