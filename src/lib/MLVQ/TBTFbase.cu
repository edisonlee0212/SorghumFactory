/*!**************************************************************************
\file    TBFbase.cpp
\author  Jiri Filip
\date    13/11/2006
\version 1.01

  The main file for the: BTFBASE project
******************************************************************************/

#include <cassert>
#include <cmath>
#include <iostream>

#include <PDF2Da.hpp>
using namespace std;

#include <AuxFuncs.hpp>

#include <cstdlib>
#include <cassert>
#include <stack>
#include <cstring>
#include <cstdio>
#include <TBTFbase.hpp>
#include <PDF2D.hpp>

// Only global variable within the project
//BTFbase *FW = 0;
// Only second global variable within the project

//int HDRflag;

//#########################################################################
//######## BFTbase ####################################################JF##
//#########################################################################
// class for pixel-wise BTF data modeling
//#########################################################################

// constructor for only loading the database
BTFbase::BTFbase(const char* basePath, bool isBRDFdata)
{
	this->m_materialOrder = 0;
	this->m_nColor = 0;

	this->m_bPath = new char[strlen(basePath) + 2];
	strcpy(this->m_bPath, (char*)basePath);

	// Sets the system variables such as discretization
	SetSystemVariables();

	return;
}//--- BTFbase ---------------------------------------------------

// Sets the material status (HDR=true or LDR=false) 
void
BTFbase::SetHdrFlag(int materialIndex, bool flagHDR, float HDRvalue)
{
	assert(m_arHdr);
	assert((materialIndex >= 0) && (materialIndex < m_materialCount));
	m_tcTemplate.m_hdrFlag = flagHDR;
	this->m_hdr = flagHDR;
	m_arHdr[materialIndex] = flagHDR;
	m_arHdRvalue[materialIndex] = HDRvalue;
} // ----------------- SetHDRflag -----------------------------------

void
BTFbase::AllocateArrays(int materialCount)
{
	// Create the arrays of PDF6D materials
	if (!m_allMaterialsInOneDatabase) {
		m_arPdf1 = new CPDF1D * [materialCount];
		assert(m_arPdf1);
		m_arAb = new CVectorColor * [materialCount];
		assert(m_arAb);
		m_arIab = new CIndexAB * [materialCount];
		assert(m_arIab);
		m_arPdf2 = new CPDF2D * [materialCount];
		assert(m_arPdf2);

		if (m_use34ViewRepresentation) {
			m_arPdf34 = new CPDF34D * [materialCount];
			assert(m_arPdf34);
			m_arPdf3 = 0; m_arPdf4 = 0;
		}
		else {
			m_arPdf3 = new CPDF3D * [materialCount];
			assert(m_arPdf3);
			m_arPdf4 = new CPDF4D * [materialCount];
			assert(m_arPdf4);
			m_arPdf34 = 0;
		}

		for (int i = 0; i < materialCount; i++) {
			m_arPdf1[i] = 0;
			m_arAb[i] = 0;
			m_arIab[i] = 0;
			m_arPdf2[i] = 0;
			if (m_use34ViewRepresentation) {
				m_arPdf34[i] = 0;
			}
			else {
				m_arPdf3[i] = 0;
				m_arPdf4[i] = 0;
			}
		} // for i
	}

	// Create the arrays of HDR flags separately for each material
	this->m_arHdr = new bool[materialCount];
	assert(this->m_arHdr);
	this->m_arHdRvalue = new float[materialCount];
	assert(this->m_arHdRvalue);
	for (int i = 0; i < materialCount; i++) {
		m_arHdr[i] = false;
		m_arHdRvalue[i] = 1.0f;
	}

	// Create the arrays of PDF6D materials, always separately
	m_arPdf6 = new CPDF6D * [materialCount];
	assert(m_arPdf6);
	m_arInputPath = new char* [materialCount]; // paths to input BTF data
	m_arOutputPath = new char* [materialCount]; // paths to output images (debugging)
	m_arTempPath = new char* [materialCount]; // paths to output images (debugging)
	m_arMaterialName = new char* [materialCount]; // names of analyzed materials

	for (int i = 0; i < materialCount; i++) {
		m_arPdf6[i] = 0;
		m_arInputPath[i] = 0;
		m_arOutputPath[i] = 0;
		m_arTempPath[i] = 0;
		m_arMaterialName[i] = 0;
	}

	return;
}//--- AllocateArrays ------------------------------------------------------

void
BTFbase::DeleteArrays()
{
	// Create the arrays of PDF6D materials
	if (!m_allMaterialsInOneDatabase) {
		delete[]m_arPdf1; m_arPdf1 = 0;
		delete[]m_arAb; m_arAb = 0;
		delete[]m_arIab; m_arIab = 0;
		if (m_use34ViewRepresentation) {
			delete[]m_arPdf34; m_arPdf34 = 0;
		}
		else {
			delete[]m_arPdf3; m_arPdf3 = 0;
			delete[]m_arPdf4; m_arPdf4 = 0;
		}
	}

	for (int i = 0; i < m_materialCount; i++) {
		if (m_arPdf6) {
			delete m_arPdf6[i]; m_arPdf6[i] = 0;
		}
		if (m_arInputPath) {
			delete[] m_arInputPath[i]; m_arInputPath[i] = 0;
		}
		if (m_arOutputPath) {
			delete[] m_arOutputPath[i]; m_arOutputPath[i] = 0;
		}
		if (m_arTempPath) {
			delete[] m_arTempPath[i]; m_arTempPath[i] = 0;
		}
		if (m_arMaterialName) {
			delete[] m_arMaterialName[i]; m_arMaterialName[i] = 0;
		}
	}

	delete[]m_arPdf6; m_arPdf6 = 0;
	delete[]m_arInputPath; m_arInputPath = 0;
	delete[]m_arOutputPath; m_arOutputPath = 0;
	delete[]m_arTempPath; m_arTempPath = 0;
	delete[]m_arMaterialName; m_arMaterialName = 0;

	return;
}//--- DeleteArrays --------------------------------------------------------

void
BTFbase::CreateSharedVariables()
{
	float* betaAngles = 0;
	float stepBeta = 0.f;
	// we always must have odd number of quantization steps per 180 degrees
	assert((this->m_lengthOfSlice % 2) == 1);
	if (m_useCosBeta) {
		printf("We use cos beta quantization with these values:\n");
		betaAngles = new float[m_lengthOfSlice];
		assert(betaAngles);
		for (int i = 0; i < m_lengthOfSlice; i++) {
			float sinBeta = -1.0f + 2.0f * i / (m_lengthOfSlice - 1);
			if (sinBeta > 1.0f)
				sinBeta = 1.0f;
			// in degrees
			betaAngles[i] = 180.f / PI * asin(sinBeta);
			printf("%3.2f ", betaAngles[i]);
		}
		printf("\n");
		betaAngles[0] = -90.f;
		betaAngles[(m_lengthOfSlice - 1) / 2] = 0.f;
		betaAngles[m_lengthOfSlice - 1] = 90.f;
	}
	else {
		// uniform quantization in angle
		printf("We use uniform angle quantization with these values:\n");
		stepBeta = 180.f / (m_lengthOfSlice - 1);
		betaAngles = new float[m_lengthOfSlice];
		assert(betaAngles);
		for (int i = 0; i < m_lengthOfSlice; i++) {
			betaAngles[i] = i * stepBeta - 90.f;
			printf("%3.2f ", betaAngles[i]);
		}
		printf("\n");
		betaAngles[(m_lengthOfSlice - 1) / 2] = 0.f;
		betaAngles[m_lengthOfSlice - 1] = 90.0f;
	}

	// Here we set alpha
	this->m_stepAlpha = 180.f / (float)(m_slicesPerHemisphere - 1);

	m_tcTemplate = TSharedCoordinates(m_useCosBeta, m_lengthOfSlice, betaAngles);

	m_tcTemplate.m_useCosBeta = this->m_useCosBeta;
	m_tcTemplate.m_lengthOfSlice = this->m_lengthOfSlice;
	// Setting alpha
	m_tcTemplate.m_stepAlpha = this->m_stepAlpha;
	m_tcTemplate.m_slicesPerHemi = this->m_slicesPerHemisphere;
	// Setting theta
	m_tcTemplate.m_slicesPerTheta = this->m_noOfTheta;
	// Setting phi
	m_tcTemplate.m_slicesPerPhi = this->m_noOfPhi;

	delete[]betaAngles; betaAngles = 0;
} // ----------- CreateSharedVariables -----------------------------

void
BTFbase::SetSystemVariables()
{
	// initial size of arrays
	this->m_maxPdf1D = 10000;
	this->m_maxVectorColor = 10000;
	this->m_maxIndexSlices = 10000;
	this->m_maxPdf2D = 10000;
	this->m_maxPdf2DLuminanceColor = 10000;
	this->m_maxPdf3D = 10000;
	this->m_maxPdf4D = 10000;
	this->m_maxPdf34D = 1000;

	// How the beta is discretized, either uniformly in degrees
	// or uniformly in cosinus of angle
	m_useCosBeta = true;

	return;
}//--- btfbase::SetSystemVariables ------------------------------------------

BTFbase::~BTFbase()
{
	for (int i = 0; i < m_materialCount; i++) {
		DeleteDatabases(i);
		DeletePaths(i);
	} // for i

	DeleteArrays();
	return;
}//--- ~BTFframework --------------------------------------------------

void
BTFbase::AllocateDatabases(int matOrder)
{
	if ((!m_allMaterialsInOneDatabase) || (matOrder == 0)) {
		m_pdf1 = new CPDF1D(m_maxPdf1D, m_lengthOfSlice, m_metric, m_maxShift);
		assert(m_pdf1);

		m_ab = new CVectorColor(m_maxVectorColor);
		assert(m_ab);

		m_iab = new CIndexAB(m_maxIndexSlices, m_lengthOfSlice, m_ab);
		assert(m_iab);
		//if (usePDF2compactRep) {
		//  PDF2 = new CPDF2Dcompact(maxPDF2D, SlicesPerHemi, PDF1, IAB, metric,
		//			       maxShift, tc, false, false);
		//    } else {
		// here we have 2D functions for luminance and color separated to individual
		// databases
		//}
		CPDF2DSeparate* PDF2Dsep =
			new CPDF2DSeparate(m_maxPdf2DLuminanceColor, m_slicesPerHemisphere, m_pdf1, m_iab, m_metric, false, false);

		m_pdf2 = PDF2Dsep;
		assert(m_pdf2);

		m_tcTemplate.m_slicesPerHemi = m_pdf2->GetSlicesPerHemisphere();

		m_pdf3 = new CPDF3D(m_maxPdf3D, m_noOfTheta, m_pdf1, m_iab, m_pdf2, m_metric);
		assert(m_pdf3);

		m_tcTemplate.m_slicesPerTheta = m_pdf3->GetSlicesPerTheta();
		m_tcTemplate.m_stepTheta = m_pdf3->GetStepTheta();

		m_pdf4 = new CPDF4D(m_maxPdf4D, m_noOfPhi, m_pdf1, m_iab, m_pdf2, m_pdf3, m_metric);
		assert(m_pdf4);

		assert(m_tcTemplate.m_slicesPerPhi == m_pdf4->GetSlicesPerPhi());
		m_tcTemplate.m_stepPhi = m_pdf4->GetStepPhi();
	}

	if (!m_allMaterialsInOneDatabase) {
		// we save the newly created instances to the arrays
		m_arPdf1[matOrder] = m_pdf1;
		m_arAb[matOrder] = m_ab;
		m_arIab[matOrder] = m_iab;
		m_arPdf2[matOrder] = m_pdf2;
		m_arPdf3[matOrder] = m_pdf3;
		m_arPdf4[matOrder] = m_pdf4;
	}

	// Class PDF6 is individual for each material even if the materials
	// are compressed using common databases (PDF1,PDF2,PDF3,PDF4,AB,IAB)
	if (m_arPdf6[matOrder]) {
		delete m_arPdf6[matOrder];
		m_arPdf6[matOrder] = 0;
	}

	m_pdf6 = new CPDF6D(m_numRows, m_numCols, m_pdf4);

	assert(m_pdf6);
	m_pdf6->SetSizes(m_noOfPhi, m_noOfTheta, m_slicesPerHemisphere,
		m_lengthOfSlice, m_nColor);
	m_pdf6->SetOffset(m_rowsOffset, m_colsOffset);

	// Set the material to the table
	m_arPdf6[matOrder] = m_pdf6;

	return;
}//--- AllocateDatabases ----------------------------------------------------------------

TileMap::~TileMap()
{
	Clear();
}

void
BTFbase::DeleteDatabases(int matOrder)
{
	if (!m_allMaterialsInOneDatabase) {
		delete m_arPdf1[matOrder]; m_arPdf1[matOrder] = 0;
		delete m_arAb[matOrder]; m_arPdf1[matOrder] = 0;
		delete m_arIab[matOrder]; m_arIab[matOrder] = 0;
		delete m_arPdf2[matOrder]; m_arPdf2[matOrder] = 0;
		delete m_arPdf3[matOrder]; m_arPdf3[matOrder] = 0;
		delete m_arPdf4[matOrder]; m_arPdf4[matOrder] = 0;
	}
	else {
		delete m_pdf1;
		delete m_ab;
		delete m_iab;
		delete m_pdf2;
		delete m_pdf3;
		delete m_pdf4;
	}
	m_pdf1 = 0;
	m_ab = 0;
	m_iab = 0;
	m_pdf2 = 0;
	m_pdf3 = 0;
	m_pdf4 = 0;

	// This is always done
	delete m_arPdf6[matOrder];
	m_arPdf6[matOrder] = 0;
	m_pdf6 = 0;

	return;
}//--- DeleteDatabases ------------------------------------------------------------------

void
BTFbase::DeletePaths(int matOrder)
{
	m_materialName = 0;
	m_iPath = 0;
	m_oPath = 0;
	m_tmpPath = 0;

	delete[] m_arMaterialName[matOrder];
	m_arMaterialName[matOrder] = 0;

	delete[] m_arInputPath[matOrder];
	m_arInputPath[matOrder] = 0;

	delete[] m_arOutputPath[matOrder];
	m_arOutputPath[matOrder] = 0;

	delete[] m_arTempPath[matOrder];
	m_arTempPath[matOrder] = 0;
} // --------- DeletePaths ------------------------------------------------------------

void
BTFbase::AllocatePaths(int matOrder, const char* nmaterialName,
	const char* niPath, const char* noPath,
	const char* ntmpPath)
{
	// set material name
	if (m_arMaterialName[matOrder])
		delete m_arMaterialName[matOrder];
	m_arMaterialName[matOrder] = new char[strlen(nmaterialName) + 2];
	strcpy(m_arMaterialName[matOrder], (char*)nmaterialName);
	this->m_materialName = m_arMaterialName[matOrder];

	// iPath
	if (m_arInputPath[matOrder])
		delete m_arInputPath[matOrder];
	m_arInputPath[matOrder] = new char[strlen(niPath) + 2];
	strcpy(m_arInputPath[matOrder], niPath);
	this->m_iPath = m_arInputPath[matOrder];
	assert(this->m_iPath);

	// oPath
	if (m_arOutputPath[matOrder])
		delete m_arOutputPath[matOrder];
	m_arOutputPath[matOrder] = new char[strlen(noPath) + 2];
	strcpy(m_arOutputPath[matOrder], noPath);
	this->m_oPath = m_arOutputPath[matOrder];
	assert(this->m_oPath);

	// tmpPath
	if (m_arTempPath[matOrder])
		delete m_arTempPath[matOrder];
	m_arTempPath[matOrder] = new char[strlen(ntmpPath) + 2];
	strcpy(m_arTempPath[matOrder], ntmpPath);
	this->m_tmpPath = m_arTempPath[matOrder];
	assert(this->m_tmpPath);

	return;
}

//! \brief sets the order of the material to be used in the application functions
int
BTFbase::SelectMaterialForRendering(int order)
{
	if ((order >= 0) && (order < m_materialCount)) {
		m_materialOrder = order;
		m_pdf6 = m_arPdf6[m_materialOrder];

		// Set proper name of the material
		m_materialName = m_arMaterialName[m_materialOrder];
		m_iPath = m_arInputPath[m_materialOrder];

		m_pdf6->GetSize(m_nRows, m_nCols);
		m_pdf6->GetOffset(m_rowsOffset, m_colsOffset);
	}

	return m_materialOrder;
}//--- selectMaterialForRendering -----------------------------------------------------

void
BTFbase::GetValRadRel(const float y,
	const float x,
	float theta_i,
	float phi_i,
	float theta_v,
	float phi_v,
	float RGB[])
{
	int xx, yy;
	if (!m_tileMap.Empty()) {
		double dummy;
		float fx = (float)modf(x, &dummy);
		float fy = (float)modf(y, &dummy);
		if (fx < 0.0f)
			fx = 1.0f + fx;
		if (fy < 0.0f)
			fy = 1.0f + fy;

		int ix = (int)(fx * m_tileMap.m_mWidth);
		int iy = (int)(fy * m_tileMap.m_mHeight);

		int pos = 2 * (ix + iy * m_tileMap.m_mWidth);
		xx = m_tileMap.m_mOffsets[pos];
		yy = m_tileMap.m_mOffsets[pos + 1];

		bool showCut = false;
		if (xx < 0 || yy < 0) {
			if (showCut) {
				// boundary of the cut
				RGB[0] = 0.f;
				RGB[1] = 0.f;
				RGB[2] = 0.f;
				return;
			}
			xx = -xx - 1;
			yy = -yy - 1;
		}
	}
	else {
		xx = x * m_numCols;
		yy = y * m_numRows;
	}
	return GetValRad(yy, xx, theta_i, phi_i, theta_v, phi_v, RGB);
}

void
BTFbase::GetValDeg(const int iRow, const int jCol, const float illuminationTheta, const float illuminationPhi,
                   const float viewTheta, const float viewPhi, float rgb[]) const
{
	if (illuminationTheta > 90.f || viewTheta > 90.f) {
		rgb[0] = 0.f;
		rgb[1] = 0.f;
		rgb[2] = 0.f;
		return;
	}

	TSharedCoordinates tc(m_tcTemplate);

#if 1
	// fast version, precomputation of interpolation values only once
	m_pdf6->GetValDeg2(iRow, jCol, illuminationTheta, illuminationPhi, viewTheta, viewPhi, rgb, tc);
#else
	// slow version
	m_pdf6->GetValDeg(iRow, jCol, illuminationTheta, illuminationPhi, viewTheta, viewPhi, rgb, tc);
#endif

	if (m_hdr) {
		// we encode the values multiplied by a user coefficient
		// before it is converted to User Color Model
		// Now we have to multiply it back.    
		const float multi = 1.0f / GetHdRvalue();
		rgb[0] *= multi;
		rgb[1] *= multi;
		rgb[2] *= multi;
	}

}//--- getValDeg ----------------------------------------------------------------

void
BTFbase::GetValRad(int irow, int jcol, float theta_i, float phi_i,
	float theta_v, float phi_v, float RGB[])
{
	if (theta_i > PI / 2.f || theta_v > PI / 2.f) {
		RGB[0] = 0.f;
		RGB[1] = 0.f;
		RGB[2] = 0.f;
		return;
	}

	TSharedCoordinates tc;
	tc = m_tcTemplate;

#if 1
	// fast version, precomputation of interpolation values only once
	m_pdf6->GetValRad2(irow, jcol, theta_i, phi_i, theta_v, phi_v, RGB, tc);
#else
	// slow version
	PDF6->getValRad(irow, jcol, theta_i, phi_i, theta_v, phi_v, RGB);
#endif  
	if (tc.m_codeBtfFlag) {
		float mul = 256.0f * m_mPostScale; ///cos(theta_i);
		//	if (HDR)
		//	  mul = (256*256.0f)/GetHDRvalue();
		RGB[0] *= mul;
		RGB[1] *= mul;
		RGB[2] *= mul;
	}
	else {
		if (m_hdr) {
			// we encode the values multiplied by a user coefficient
			// before it is converted to User Color Model
			// Now we have to multiply it back.    
			//    float mult = 1.0f/GetHDRvalue();
			float mult = 255.0f / GetHdRvalue();
			RGB[0] *= mult;
			RGB[1] *= mult;
			RGB[2] *= mult;
		}
	}
}//--- getValRad ----------------------------------------------------------------


void
BTFbase::GetValDegShepard(int irow, int jcol, float theta_i, float phi_i,
	float theta_v, float phi_v, float RGB[])
{
	if (theta_i > 90.f || theta_v > 90.f) {
		RGB[0] = 0.f;
		RGB[1] = 0.f;
		RGB[2] = 0.f;
		return;
	}

	TSharedCoordinates tc(m_tcTemplate);

	// fast version, precomputation of interpolation values only once
	m_pdf6->GetValDegShepard(irow, jcol, theta_i, phi_i, theta_v, phi_v, RGB, tc);

	if (m_hdr) {
		// we encode the values multiplied by a user coefficient
		// before it is converted to User Color Model
		// Now we have to multiply it back.    
		float mult = 1.0f / GetHdRvalue();
		RGB[0] *= mult;
		RGB[1] *= mult;
		RGB[2] *= mult;
	}

	return;
}//--- getValDegShepard ------------------------------------------------------------

void
BTFbase::GetValDegShepard2(int irow, int jcol, float theta_i, float phi_i,
	float theta_v, float phi_v, float RGB[])
{
	if (theta_i > 90.f || theta_v > 90.f) {
		RGB[0] = 0.f;
		RGB[1] = 0.f;
		RGB[2] = 0.f;
		return;
	}

	TSharedCoordinates tc(m_tcTemplate);

	// fast version, precomputation of interpolation values only once
	m_pdf6->GetValDegShepard2(irow, jcol, theta_i, phi_i, theta_v, phi_v, RGB, tc);

	if (m_hdr) {
		// we encode the values multiplied by a user coefficient
		// before it is converted to User Color Model
		// Now we have to multiply it back.    
		float mult = 1.0f / GetHdRvalue();
		RGB[0] *= mult;
		RGB[1] *= mult;
		RGB[2] *= mult;
	}

	return;
}//--- getValDegShepard2 ------------------------------------------------------------

bool
BTFbase::LoadBtfbase(const char* prefix, bool recover)
{
	// saving individual databases
	char fileName[1000];

	// try to load btf tile map
	{
		printf("Loading tile map...");
		sprintf(fileName, "%s%stile_map.txt", m_bPath, prefix);
		char buffer[256];
		FILE* fp;
		if ((fp = fopen(fileName, "rt")) != NULL) {
			char* vv = fgets(buffer, 256, fp); vv = vv;
			assert(vv);
			if (sscanf(buffer, "%d %d", &m_tileMap.m_mWidth, &m_tileMap.m_mHeight) == 2) {
				m_tileMap.m_mOffsets = new int[2 * m_tileMap.m_mWidth * m_tileMap.m_mHeight];
				//int index=0;
				cout << "tileMap " << m_tileMap.m_mWidth << "x" << m_tileMap.m_mHeight << endl;
				int loadEntries = 2 * m_tileMap.m_mWidth * m_tileMap.m_mHeight;
				for (int i = 0; i < loadEntries; ) {
					int x, y;
					if (fgets(buffer, 256, fp) == NULL || (sscanf(buffer, "%d %d", &x, &y) != 2)) {
						printf("Error in loading tile map %d %d\n", i, loadEntries);
						m_tileMap.Clear();
						break;
					}
					m_tileMap.m_mOffsets[i++] = x;
					m_tileMap.m_mOffsets[i++] = y;
					//		  cout<<i<<" "<<x<<" "<<y<<" , ";
				}
			}
		}

		if (!m_tileMap.Empty()) {
			printf("Successfully loaded tile map\n");
		}
		else {
			// exit(0);
		}
	}
	m_mPostScale = 1.0f;
	{
		printf("Loading scale info...");
		sprintf(fileName, "%s%sscale.txt", m_bPath, prefix);
		char buffer[256];
		FILE* fp;
		if ((fp = fopen(fileName, "rt")) != NULL) {
			char* vv = fgets(buffer, 256, fp); vv = vv;
			assert(vv);
			if (sscanf(buffer, "%f", &m_mPostScale) != 1)
				m_mPostScale = 1.0f;
		}
	}
	// -----------------------------------------------------
	// Saving common informations
	sprintf(fileName, "%s%sall_materialInfo.txt", m_bPath, prefix);

	FILE* fp;
	if ((fp = fopen(fileName, "r")) == NULL) {
		printf("Error - opening file %s !!!\n", fileName);
		return true;
	}
	// First save the info about BTFbase: name, materials saved, and how saved
	char line[1000];
	int loadMaterials;
	int maxMaterials;
	int flagAllMaterials;
	int flagUse34DviewRep;
	int flagUsePDF2compactRep;

	// First save the info about BTFbase: name, materials saved, and how saved
	if (fscanf(fp, "%s\n%d\n%d\n%d\n%d\n%d\n", &(line[0]), &loadMaterials, &maxMaterials,
		&flagAllMaterials, &flagUse34DviewRep, &flagUsePDF2compactRep) != 6) {
		fclose(fp);
		printf("File is corrupted for reading basic parameters\n");
		exit(-1);
	}
	// Here we need to read this information about original data
	int ncolour, nview, nillu, tileSize;
	if (fscanf(fp, "%d\n%d\n%d\n%d\n", &ncolour, &nview, &nillu, &tileSize) != 4) {
		fclose(fp);
		printf("File is corrupted for reading basic parameters about orig database\n");
		exit(-1);
	}

	// Here we load how parameterization is done
	// It is meant: beta/stepPerBeta, alpha/stepsPerAlpha, theta/stepsPerTheta, phi/stepPerPhi,
	// reserve/reserv, reserve/reserve
	int useCosBetaFlag, stepsPerBeta, tmp3, stepsPerAlpha, tmp5, stepsPerTheta,
		tmp7, stepsPerPhi, tmp9, tmp10, tmp11, tmp12;
	if (fscanf(fp, "%d %d %d %d %d %d %d %d %d %d %d %d\n", &useCosBetaFlag, &stepsPerBeta, &tmp3,
		&stepsPerAlpha, &tmp5, &stepsPerTheta, &tmp7, &stepsPerPhi, &tmp9,
		&tmp10, &tmp11, &tmp12) != 12) {
		fclose(fp);
		printf("File is corrupted for reading angle parameterization settings\n");
		exit(-1);
	}
	m_useCosBeta = useCosBetaFlag ? true : false;
	m_lengthOfSlice = stepsPerBeta;
	assert((m_lengthOfSlice % 2) == 1);
	m_slicesPerHemisphere = stepsPerAlpha;
	assert((m_slicesPerHemisphere % 2) == 1);
	m_noOfTheta = stepsPerTheta;
	assert(m_noOfTheta >= 2);
	m_noOfPhi = stepsPerPhi;
	assert(m_noOfPhi >= 1);

	// Here we recreate shared variables (angles parameterization)
	CreateSharedVariables();

	// use a specific flag when processing data from code BTF
	m_tcTemplate.m_codeBtfFlag = tmp12;

	// Here we need to read this information about current material setting
	// where are the starting points for the next search, possibly
	int fPDF1, fAB, fIAB, fPDF2, fPDF2L, fPDF2AB, fPDF3, fPDF34, fPDF4, fRESERVE;
	if (fscanf(fp, "%d %d %d %d %d %d %d %d %d %d\n",
		&fPDF1, &fAB, &fIAB, &fPDF2, &fPDF2L,
		&fPDF2AB, &fPDF3, &fPDF34, &fPDF4, &fRESERVE) != 10) {
		fclose(fp);
		printf("File is corrupted for reading starting search settings\n");
		exit(-1);
	}
	// Here we need to save this information about current material setting
	int lsPDF1, lsAB, lsIAB, lsPDF2, lsPDF2L, lsPDF2AB, lsPDF3, lsPDF34, lsPDF4, lsRESERVE;
	if (fscanf(fp, "%d %d %d %d %d %d %d %d %d %d\n",
		&lsPDF1, &lsAB, &lsIAB, &lsPDF2, &lsPDF2L,
		&lsPDF2AB, &lsPDF3, &lsPDF34, &lsPDF4, &lsRESERVE) != 10) {
		fclose(fp);
		printf("File is corrupted for reading starting search points\n");
		exit(-1);
	}

	int metric;
	float baseEps, rPDF1, epsAB, epsIAB, rPDF2, rPDF2L, epsPDF2AB, rPDF3, rPDF34, rPDF4, rPDF4b;
	if (fscanf(fp, "%d %f %f %f %f %f %f %f %f %f %f %f\n", &metric, &baseEps,
		&rPDF1, &epsAB, &epsIAB, &rPDF2, &rPDF2L,
		&epsPDF2AB, &rPDF3, &rPDF34, &rPDF4, &rPDF4b) != 12) {
		fclose(fp);
		printf("File is corrupted for reading epsilon search settings\n");
		exit(-1);
	}

	// !!!!!! If we have only one database for all materials or
	// we share some databases except PDF6 for all materials
	this->m_allMaterialsInOneDatabase = flagAllMaterials;
	this->m_use34ViewRepresentation = flagUse34DviewRep;
	this->m_usePdf2CompactRep = flagUsePDF2compactRep;

	if (loadMaterials > maxMaterials)
		loadMaterials = maxMaterials;
	m_materialCount = maxMaterials;
	if (flagAllMaterials) {
		m_allMaterialsInOneDatabase = true;
		printf("Loading all materials from one database\n");
	}
	else {
		m_allMaterialsInOneDatabase = false;
		printf("Loading materials from several separate databases\n");
	}
	// Here we need to allocate the structures
	AllocateArrays(m_materialCount);

	// Now we read line by line the names of the materials etc
	int i;
	for (i = 0; i < loadMaterials; i++) {
		int ro, co, pr, pc;
		char l1[1000], l2[1000], l3[1000], l4[1000];
		int hdrFlag = 0;
		float hdrValue;
		if (fscanf(fp, "%s %s %s %s %d %d %d %d %f\n", l1, l2, l3, l4, &ro, &co, &pr, &pc, &hdrValue) == 9)
		{
			// Here we need to allocate the arrays for names
			m_arMaterialName[i] = new char[strlen(l1) + 30];
			strcpy(m_arMaterialName[i], l1);
			m_arInputPath[i] = new char[strlen(l2) + 30];
			strcpy(m_arInputPath[i], l2);
			m_arOutputPath[i] = new char[strlen(l3) + 30];
			strcpy(m_arOutputPath[i], l3);
			m_arTempPath[i] = new char[strlen(l4) + 30];
			strcpy(m_arTempPath[i], l4);

			printf("Loading material will try for: %s %s %s %s %d %d %d %d %f\n", m_arMaterialName[i],
				m_arInputPath[i], m_arOutputPath[i], m_arTempPath[i], ro, co, pr, pc, hdrValue);

			if ((fabs(hdrValue - 1.0f) < 1e-6) ||
				(fabs(hdrValue) < 1e-6)) {
				hdrFlag = 0;
				hdrValue = m_arHdRvalue[m_materialOrder] = 1.0f;
			}
			else {
				hdrFlag = 1;
				m_arHdRvalue[m_materialOrder] = hdrValue;
			}

			SetHdrFlag(i, hdrFlag, hdrValue);
		}
		else {
			loadMaterials = i;
			break;
		}
	} // for i
	fclose(fp);

	// Note that nrows and ncols are not set during loading !
	const int algPut = 0;

	if (m_allMaterialsInOneDatabase) {
		// Now creating PDF6 for each material using common database
		printf("Loading materials for common DBF1,DBF2,DBF3,DBF4,AB,IAB database\n");
		for (i = 0; i < loadMaterials; i++) {
			sprintf(fileName, "%s%s%s_materialInfo.txt", m_bPath, prefix, m_arMaterialName[i]);
			FILE* fp;
			if ((fp = fopen(fileName, "r")) == NULL) {
				printf("Error - opening file %s !!!\n", fileName);
				return true;
			}
			int ro, co, pr, pc, hdrf;
			float hdrvalue;
			char nameM[200];
			if (fscanf(fp, "%s %s %s %s %d %d %d %d %f\n", &(nameM[0]),
				&(m_arInputPath[i][0]), &(m_arOutputPath[i][0]), &(m_arTempPath[i][0]), &ro, &co, &pr, &pc, &hdrvalue)
				!= 9) {
				printf("ERROR:Reading the information about material index=%d failed\n", i);
				printf("Exiting\n");
				fclose(fp); exit(-1);
			}
			fclose(fp);
			if ((fabs(hdrvalue - 1.0f) < 1e-6) ||
				(fabs(hdrvalue) < 1e-6)) {
				hdrf = 0;
				hdrvalue = m_arHdRvalue[m_materialOrder] = 1.0f;
			}
			else {
				hdrf = 1;
				m_arHdRvalue[m_materialOrder] = hdrvalue;
			}

			if (strcmp(nameM, m_arMaterialName[i]) != 0) {
				printf("Some problem material name in file=%s other name=%s\n",
					nameM, m_arMaterialName[i]);
				AbortRun(1232);
			}

			// Now we can create the database, PDF6 is allocated
			// with right values
			m_numRows = pr; m_numCols = pc;
			m_rowsOffset = ro; m_colsOffset = co;
			SetHdrFlag(i, hdrf, hdrvalue);
			printf("Setting HDRF=%d\n", hdrf);

			// Allocate the databases PDF1,AB,IAB,PDF2,PDF3,PDF4,PDF6
			AllocateDatabases(i);

			// loading individual databases for PDF6 that are not shared
			sprintf(fileName, "%s%s%s", m_bPath, prefix, m_arMaterialName[i]);
			assert(m_arPdf6[i]);
			if (m_arPdf6[i]->Load(fileName, 0)) {
				printf("ERROR: loading of PDF6 has failed\n");
				// PDF6 was not created and it cannot be loaded then
				return true; // error
			}
			// We also need to set offset
			m_arPdf6[i]->SetOffset(m_rowsOffset, m_colsOffset);
		}
		// PDF6 databases were loaded, now we load the rest
		sprintf(fileName, "%s%sall", m_bPath, prefix);
		// loading individual databases except PDF6 that is already loaded
		if (m_pdf1->Load(fileName, 0, algPut)) {
			printf("ERROR: loading of PDF1 has failed\n");
			return true; // error
		}
		if (m_ab->Load(fileName, 0, algPut)) {
			printf("ERROR: loading of AB has failed\n");
			return true; // error
		}
		if (m_iab->Load(fileName, 0, algPut)) {
			printf("ERROR: loading of IAB has failed\n");
			return true; // error
		}

		if (m_pdf2->Load(fileName, 0, m_pdf1->GetNumOfPdf1D(),
			m_iab->GetNoOfIndexSlices(), algPut)) {
			printf("ERROR: loading of PDF2 has failed\n");
			return true; // error
		}
		if (m_pdf3->Load(fileName, 0, m_pdf2->GetNoOfPdf2D(), algPut)) {
			printf("ERROR: loading of PDF3 has failed\n");
			return true; // error
		}
		if (m_pdf4->Load(fileName, 0, m_pdf3->GetNoOfPdf3D(), algPut)) {
			printf("ERROR: loading of PDF4 has failed\n");
			return true; // error
		}
		printf("The database was read successfully\n");
		m_materialName = m_arMaterialName[0];
		return false; // OK - database loaded, or at least partially
	}

	// Load each material separately
	for (i = 0; i < loadMaterials; i++) {
		sprintf(fileName, "%s%s%s_materialInfo.txt", m_bPath, prefix, m_arMaterialName[i]);
		FILE* fp;
		if ((fp = fopen(fileName, "r")) == NULL) {
			printf("Error - opening file %s !!!\n", fileName);
			return true;
		}
		int ro, co, pr, pc;
		float hdrValue;
		if (fscanf(fp, "%s %s %s %s %d %d %d %d %f\n", &(m_arMaterialName[i][0]),
			&(m_arInputPath[i][0]), &(m_arOutputPath[i][0]), &(m_arTempPath[i][0]),
			&ro, &co, &pr, &pc, &hdrValue) != 9) {
			printf("Error - reading parameters from file %s failed\n", fileName);
			return true;
		}

		if ((fabs(hdrValue - 1.0f) < 1e-6) ||
			(fabs(hdrValue) < 1e-6)) {
			m_arHdr[i] = false;
			hdrValue = m_arHdRvalue[i] = 1.0f;
		}
		else {
			m_arHdr[i] = true;
			m_arHdRvalue[i] = hdrValue;
		}

		fclose(fp);
		// Now we can create the database, PDF6 is allocated
		// with right values
		m_numRows = pr; m_numCols = pc;
		m_rowsOffset = ro; m_colsOffset = co;

		// -----------
		// Allocate the databases PDF1,AB,IAB,PDF2,PDF3,PDF4,PDF6
		AllocateDatabases(i);

		sprintf(fileName, "%s%s%s", m_bPath, prefix, m_arMaterialName[i]);
		if (!m_arPdf1[i]) {
			printf("ERROR: loading of PDF1 has failed - instance not created\n");
			return true;
		}
		if (m_arPdf1[i]->Load(fileName, 0, algPut)) {
			if (recover) break;
			printf("ERROR: loading of PDF1 has failed\n");
			return true;
		}
		if (!m_arAb[i]) return true;
		if (m_arAb[i]->Load(fileName, 0, algPut)) {
			if (recover) break;
			printf("ERROR: loading of AB has failed\n");
			return true;
		}
		if (!m_arIab[i]) return true;
		if (m_arIab[i]->Load(fileName, 0, algPut)) {
			if (recover) break;
			printf("ERROR: loading of IAB has failed\n");
			return true;
		}
		if (!m_arPdf2[i]) return true;
		if (m_arPdf2[i]->Load(fileName, 0, m_arPdf1[i]->GetNumOfPdf1D(),
			m_arIab[i]->GetNoOfIndexSlices(), algPut)) {
			if (recover) break;
			printf("ERROR: loading of PDF2 has failed\n");
			return true;
		}
		if (!m_arPdf3[i]) return true;
		if (m_arPdf3[i]->Load(fileName, 0, m_pdf2->GetNoOfPdf2D(), algPut)) {
			if (recover) break;
			printf("ERROR: loading of PDF3 has failed\n");
			return true;
		}
		if (!m_arPdf4[i]) return true;
		if (m_arPdf4[i]->Load(fileName, 0, m_pdf3->GetNoOfPdf3D(), algPut)) {
			if (recover) break;
			printf("ERROR: loading of PDF4 has failed\n");
			return true;
		}

		if (!m_arPdf6[i]) return true;
		if (m_arPdf6[i]->Load(fileName, 0)) {
			if (recover) break;
			printf("ERROR: loading of PDF6 has failed\n");
			return true;
		}
		// We also need to set offset
		m_arPdf6[i]->SetOffset(m_rowsOffset, m_colsOffset);

		printf("Material loaded: %s %s %s %s %d %d %d %d\n", m_arMaterialName[i],
			m_arInputPath[i], m_arOutputPath[i], m_arTempPath[i], ro, co, pr, pc); fflush(stdout);
	} // for i


	printf("Loading of all %d materials is finished\n", i); fflush(stdout);

	printf("The database was read successfully (separate materials)\n");
	fflush(stdout);
	m_materialName = m_arMaterialName[0];
	return false; // ok, databases were loaded separately
}//---- loadBTFbase----------------------------------------------------------------------

// Here we compute the size of the database 
void
BTFbase::ComputeSizes(
	// input
	bool onlyCurrentMaterial,
	int matOrderCurrent,
	int cntPixelsCurrent,
	// output
	long double& origSize,
	long double& compSize,
	long double& compSizeQuantized,
	// input ... compute quantized version - it can take time
	bool cq)
{
	// New size is computed in long ints, this should be enough, when the
	// material is compressed
	unsigned long int newSize = 0UL;
	unsigned long int newSizeQ = 0UL;

	if (onlyCurrentMaterial || m_allMaterialsInOneDatabase) {
		newSize += (unsigned long int)m_pdf1->GetMemory();
		newSize += (unsigned long int)m_ab->GetMemory();
		newSize += (unsigned long int)m_iab->GetMemory();
		newSize += (unsigned long int)m_pdf2->GetMemory();

		if (cq) {
			newSizeQ += (unsigned long int)m_pdf1->GetMemoryQ();
			newSizeQ += (unsigned long int)m_ab->GetMemoryQ();
			newSizeQ += (unsigned long int)m_iab->GetMemoryQ();
			newSizeQ += (unsigned long int)m_pdf2->GetMemoryQ();
		}

		newSize += (unsigned long int)(m_pdf3->GetMemory());
		newSize += (unsigned long int)(m_pdf4->GetMemory());
		if (cq) newSizeQ += (unsigned long int)(m_pdf3->GetMemoryQ());
		if (cq) newSizeQ += (unsigned long int)(m_pdf4->GetMemoryQ());
		CPDF2DSeparate* pdf2s = dynamic_cast<CPDF2DSeparate*>(m_pdf2);
		newSize += (unsigned long int)(pdf2s->GetMemoryLuminance() + pdf2s->GetMemoryColor());
		if (cq) newSizeQ += (unsigned long int)(pdf2s->GetMemoryLuminanceQ() + pdf2s->GetMemoryColorQ());
	}
	else {
		int i;
		for (i = 0; i <= matOrderCurrent; i++) {
			newSize += (unsigned long int)m_arPdf1[i]->GetMemory();
			newSize += (unsigned long int)m_arAb[i]->GetMemory();
			newSize += (unsigned long int)m_arIab[i]->GetMemory();
			newSize += (unsigned long int)m_arPdf2[i]->GetMemory();

			if (cq) {
				newSizeQ += (unsigned long int)m_arPdf1[i]->GetMemoryQ();
				newSizeQ += (unsigned long int)m_arAb[i]->GetMemoryQ();
				newSizeQ += (unsigned long int)m_arIab[i]->GetMemoryQ();
				newSizeQ += (unsigned long int)m_arPdf2[i]->GetMemoryQ();
			}

			newSize += (unsigned long int)(m_arPdf3[i]->GetMemory());
			newSize += (unsigned long int)(m_arPdf4[i]->GetMemory());
			if (cq) {
				newSizeQ += (unsigned long int)(m_arPdf3[i]->GetMemoryQ());
				newSizeQ += (unsigned long int)(m_arPdf4[i]->GetMemoryQ());
			}
			CPDF2DSeparate* pdf2s = dynamic_cast<CPDF2DSeparate*>(m_arPdf2[i]);
			newSize += (unsigned long int)(pdf2s->GetMemoryLuminance());
			newSize += (unsigned long int)(pdf2s->GetMemoryColor());
			if (cq) {
				newSizeQ += (unsigned long int)(pdf2s->GetMemoryLuminanceQ());
				newSizeQ += (unsigned long int)(pdf2s->GetMemoryColorQ());
			}
		} // for
	}

	// This is now corrected to be computed in long doubles for sure.
	// For HDR we take 12 bits per each channel, for 8-bits PNG data we take 8 bits
	// per one value.

	// Here we compute the number of input data for UBO representation
	const int nillu = 81; const int nview = 81;
	const int ncolour = 2;
	origSize = (long double)(nillu * nview * (ncolour + 1));
	if (m_arHdr[matOrderCurrent])
		// Here we take 12 bits instead of 8 bits, as HDR can be efficiently
		// compressed with 12 bits in logarithmic scale. Question is whether
		// this is fair or too strict.
		origSize *= 1.5f;
	unsigned long int cntPixelsOrig = 0;

	// Add the memory for the indices in PDF6
	int i;
	for (i = 0; i <= matOrderCurrent; i++) {
		if (i != matOrderCurrent) {
			// we assume that all pixels in the material are compressed
			newSize += (unsigned long int)(m_arPdf6[i]->GetMemory());
			cntPixelsOrig += m_arPdf6[i]->GetCountPixels();
			if (cq) newSizeQ += (unsigned long int)(m_arPdf6[i]->GetMemoryQ());
		}
		else {
			// only processed pixels
			newSize += (unsigned long int)(m_arPdf6[i]->GetMemory());
			cntPixelsOrig += cntPixelsCurrent;
			if (cq) newSizeQ += (unsigned long int)(m_arPdf6[i]->GetMemoryQ());
		}
	} // for i

	// The size in bytes
	origSize *= cntPixelsOrig;
	// The size without compression of indices
	compSize = (long double)newSize;
	// The size with compression of indices requiring only necessary number of bits
	compSizeQuantized = (long double)newSizeQ;

	return;
} // ----------------------------------------------------------------------


// !\ brief implements importance sampling. It given random variables in range [0-1]x[0-1]
// it generates a ray along this direction.
int
BTFbase::ImportanceSamplingDeg(int irow, int jcol, float theta_v, float phi_v,
	float q0, float q1, float& theta_i, float& phi_i)
{
	if (theta_v > 90.f)
		return 1; // no meaning, out of range, we have failed

	TSharedCoordinates tc(m_tcTemplate);

	// fast version, precomputation of interpolation values only once
	return m_pdf6->ImportanceSamplingDeg(irow, jcol, theta_v, phi_v, q0, q1, theta_i, phi_i, tc);
}


// !\ brief implements importance sampling. It given random variables in range [0-1]x[0-1]
// it generates a ray along this direction.
int
BTFbase::ImportanceSamplingDeg(const int iRow, const int jCol, const float viewTheta, const float viewPhi,
                               const int cntRays, float q0Q1[], float illuminationThetaPhi[]) const
{
	if (viewTheta >= 90.f)
		return 0; // no meaning, out of range, we have failed for any ray

	TSharedCoordinates tc(m_tcTemplate);

	// fast version, pre-computation of interpolation values only once
	return m_pdf6->ImportanceSamplingDeg(iRow, jCol, viewTheta, viewPhi, cntRays, q0Q1, illuminationThetaPhi, tc);
}


//! \brief computes albedo for fixed viewer direction
void
BTFbase::GetViewerAlbedoDeg(int irow, int jcol, float theta_v, float phi_v, float RGB[])
{
	if (theta_v >= 90.f) {
		RGB[0] = RGB[1] = RGB[2] = 0.f;
		return;
	}

	TSharedCoordinates tc(m_tcTemplate);
	m_pdf6->GetViewerAlbedoDeg(irow, jcol, theta_v, phi_v, RGB, tc);

	if (m_hdr) {
		// we encode the values multiplied by a user coefficient
		// before it is converted to User Color Model
		// Now we have to multiply it back.    
		float mult = 1.0f / GetHdRvalue();
		RGB[0] *= mult;
		RGB[1] *= mult;
		RGB[2] *= mult;
	}

	return;
}


// This given the indices to new onion parametrization computes the theta,phi angles
// for light direction (theta_i, phi_i) and viewer direction (theta_v, phi_v)
int
BTFbase::GetParametrizationDegBtfBase(int indexBeta, int indexAlpha, int indexTheta, int indexPhi,
	float& theta_i, float& phi_i, float& theta_v, float& phi_v,
	TSharedCoordinates& tc)
{
	if ((indexBeta < 0) || (indexBeta >= m_lengthOfSlice))
		return 1;
	if ((indexAlpha < 0) || (indexAlpha >= m_slicesPerHemisphere))
		return 1;
	if ((indexTheta < 0) || (indexTheta >= m_noOfTheta))
		return 1;
	if ((indexPhi < 0) || (indexPhi >= m_noOfPhi))
		return 1;

	float deltaTheta = 90.f / (float)(this->m_noOfTheta - 1); //for 7 we get 15 degrees
	float deltaPhi = 360.f / (float)this->m_noOfPhi; // for 24 we get 15 degrees

	// Viewer direction
	phi_v = (float)(indexPhi * deltaPhi);

	// recompute from clockwise to anti-clockwise phi_v notation: (360.f - phi_v) 
	phi_v = 360.f - phi_v;
	while (phi_v >= 360.f)
		phi_v -= 360.f;
	while (phi_v < 0.f)
		phi_v += 360.f;

	theta_v = (float)(indexTheta * deltaTheta);

	// Illumination direction

	float alpha = tc.ComputeAngleAlpha(indexAlpha, 0);
	alpha *= PI / 180.f;
	float beta = tc.ComputeAngleBeta(indexBeta, 0);
	beta *= PI / 180.f;
	ConvertBetaAlphaToThetaPhi(beta, alpha, theta_i, phi_i);
	theta_i *= 180.f / PI;
	phi_i *= 180.f / PI;

	// rotation of onion-sliced parametrization to be perpendicular to phi_v of
	// viewing angle: - (90.f + phi_v)
	phi_i = phi_i + phi_v + 90.f;

	while (phi_i >= 360.f)
		phi_i -= 360.f;
	while (phi_i < 0.f)
		phi_i += 360.f;

	return 0;
}
