/*!**************************************************************************
\file    IndexAB.cpp
\author  V. Havran
\date    15/12/2010
\version 0.00
*/

#include <cstdio>
#include <TBTFbase.hpp>

static BTFbase* BTF = 0;

void
SetBtfBasePointer(void* newPointer)
{
	BTF = (BTFbase*)newPointer;
}

void*
GetBtfBasePointer()
{
	return (void*)BTF;
}

int
InitBtfBase(const char* parName)
{
	//## DBFBASE #######################################
	// UBO BTF data specific parameters (for all of them)

	// reading parametric file
	FILE* fp;

	// we have to find out, if this is par file or the directory
	char fileName[1000];
	sprintf(fileName, "%s/all_materialInfo.txt", parName);
	FILE* fp2;

	bool avoidParFile = false;
	if ((fp2 = fopen(fileName, "r")) != NULL) {
		printf("BTFbase is read from the directory %s\n", parName);
		fclose(fp2);
		avoidParFile = true;
	}

	if (!avoidParFile) {
		if ((fp = fopen(parName, "r")) == NULL) {
			printf("Parametric file is wrong or was not specified!\n\n");
			return 1; // we cannot read the database
		}
		printf("BTFbase is read according to paramteric file %s\n", parName);
	}

	char cd;
	char btfbasePath[1000];
	int countMaterials;
	if (!avoidParFile) {
		printf("_________________________________________\n");
		printf("Loading BTFBASE accorcing parametric file\n");
		printf("*****************************************\n");

		// number of materials to be compresses
		int v = fscanf(fp, "%d\n", &countMaterials); v = v;
		assert(v == 1);
		do { cd = getc(fp); } while (cd != '\n');
		v = fscanf(fp, "%s\n", btfbasePath);
		assert(v == 1);
		do { cd = getc(fp); } while (cd != '\n');
		printf("Using BTFBase Path: '%s'\n", btfbasePath);
	}
	else {
		printf("Loading BTFBASE from the directory %s\n", fileName);
		sprintf(btfbasePath, "%s/", parName);
	}

	// we will load the data
	BTF = 0;

	// For matusik it should be true here !
	//bool isBRDFdata = false;

	BTF = new BTFbase(btfbasePath, false);
	assert(BTF);
	BTF->LoadBtfbase("", false);

	if (!avoidParFile)
		fclose(fp);

	printf("BTFBAse was successfully loaded according to %s\n", parName);
	return 0; // OK, success
}//--- initBTFBASE -----------------------------------------------------------------------------

void
FinishBtfBase()
{
	delete BTF;
	BTF = 0;
}//--- finishBTFBASE -------------------------------------------------------------------------------

void
GetSizeBtfBase(int& rowsNum, int& colsNum)
{
	BTF->GetSize(rowsNum, colsNum);
}

void
GetBtfDeg(const int iRow, const int jCol, const float illuminationTheta, const float illuminationPhi,
          const float viewTheta, const float viewPhi, float rgb[])
{
	assert(BTF);
	const float phiI2 = -illuminationPhi;
	const float phiV2 = -viewPhi;
#if 1
	BTF->GetValDeg(iRow, jCol, illuminationTheta, phiI2, viewTheta, phiV2, rgb);
#else  
	//BTF->GetValDegShepard(iRow, jCol, illuminationTheta, phiI2, viewTheta, phiV2, rgb);
	BTF->GetValDegShepard2(iRow, jCol, illuminationTheta, phiI2, viewTheta, phiV2, rgb);
#endif  
}//--- getValDeg -----------------------------------------------------------------------------

void
GetBtfRad(int iRow, int jCol, float illuminationTheta, float illuminationPhi,
	float viewTheta, float viewPhi, float rgb[])
{
	assert(BTF);

	float phi_i2 = -illuminationPhi;
	float phi_v2 = -viewPhi;

	BTF->GetValRad(iRow, jCol, illuminationTheta, phi_i2, viewTheta, phi_v2, rgb);
}//--- getValRad -----------------------------------------------------------------------------

void
SelectBtfMaterial(const int materialIndex)
{
	assert(BTF);
	BTF->SelectMaterialForRendering(materialIndex);
}

int
GetIndexOfSelectedBtfMaterial()
{
	assert(BTF);
	return BTF->GetMaterialOrder();
}

const char*
GetBtfMaterialName()
{
	assert(BTF);
	return BTF->GetCurrentMaterialName();
}

const char*
GetBtfMaterialName(int materialIndex)
{
	if ((materialIndex < 0) || (materialIndex >= BTF->GetMaterialCount())) {
		printf("ERROR: Illegal index of the material\n");
		return 0; //
	}

	assert(BTF);
	int tmp = BTF->GetMaterialOrder();
	BTF->SelectMaterialForRendering(materialIndex);
	const char* name = BTF->GetCurrentMaterialName();
	BTF->SelectMaterialForRendering(tmp);
	return name;
}

int
GetBtfMaterialCount()
{
	assert(BTF);
	// Number of compressed materials in this database
	return BTF->GetMaterialCount();
}

// Make importance sampling
int
ImportanceSamplingDegBtfBase(int iRow, int jCol, float viewTheta, float viewPhi,
	float q0, float q1, float& illuminationTheta, float& illuminationPhi)
{
	assert(BTF);
	return BTF->ImportanceSamplingDeg(iRow, jCol, viewTheta, viewPhi,
		q0, q1, illuminationTheta, illuminationPhi);
}

int
ImportanceSamplingDegBtfBase(const int iRow, const int jCol, const float viewTheta, const float viewPhi,
                             const int cntRays, float q0Q1[], float illuminationThetaPhi[])
{
	assert(BTF);
	return BTF->ImportanceSamplingDeg(iRow, jCol, viewTheta, viewPhi,
		cntRays, q0Q1, illuminationThetaPhi);
}

void
GetViewerAlbedoDegBtfBase(int iRow, int jCol, float viewTheta, float viewPhi, float rgb[])
{
	assert(BTF);
	BTF->GetViewerAlbedoDeg(iRow, jCol, viewTheta, viewPhi, rgb);
}


// Given the angle indices in the new onion parametrization, it returns the
// light direction theta_i, phi_i and viewer direction theta_v, phi_v in
// degrees.
int GetParametrizationDegBtfBase(int indexBeta, int indexAlpha, int indexTheta, int indexPhi,
	float& illuminationTheta, float& illuminationPhi, float& viewTheta, float& viewPhi)
{
	assert(BTF);

	// $$JB - warning this is not correct, but not used anyway!!!
	TSharedCoordinates dummy;
	return BTF->GetParametrizationDegBtfBase(indexBeta, indexAlpha, indexTheta, indexPhi,
		illuminationTheta, illuminationPhi, viewTheta, viewPhi, dummy);
}
