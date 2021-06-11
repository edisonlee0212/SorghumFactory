#include <BTFIAB.cuh>
#include <functional>
#include <exception>
#include <fstream>
#include <filesystem>
#include <FileIO.hpp>
#include <mutex>
#include <Debug.hpp>
using namespace RayTracerFacility;
using namespace UniEngine;

bool ParseFloatData(const std::string& fileName, int& numOfRows, int& numOfCols, float& minValue, float& maxValue, std::vector<float>& data)
{
	FILE* fp;
	if ((fp = fopen(fileName.c_str(), "r")) == nullptr) {
		UNIENGINE_ERROR("Error");
		return false;
	}
	int v = fscanf(fp, "%d %d %f %f\n", &numOfRows, &numOfCols, &minValue, &maxValue);
	assert(v == 4);
	data.resize(numOfCols * numOfRows);
	for (int row = 0; row < numOfRows; row++) {
		for (int col = 0; col < numOfCols; col++) {
			v = fscanf(fp, "%f ", &data[row * numOfCols + col]);
			assert(v == 1);
		}
		fscanf(fp, "\n");
	}
	fclose(fp);
	return true;
}

bool ParseIntData(const std::string& fileName, int& numOfRows, int& numOfCols, int& minValue, int& maxValue, std::vector<int>& data)
{
	FILE* fp;
	if ((fp = fopen(fileName.c_str(), "r")) == nullptr) {
		UNIENGINE_ERROR("Error");
		return false;
	}
	int v = fscanf(fp, "%d %d %d %d\n", &numOfRows, &numOfCols, &minValue, &maxValue);
	assert(v == 4);
	data.resize(numOfCols * numOfRows);
	for (int row = 0; row < numOfRows; row++) {
		for (int col = 0; col < numOfCols; col++) {
			v = fscanf(fp, "%d ", &data[row * numOfCols + col]);
			assert(v == 1);
		}
		fscanf(fp, "\n");
	}
	fclose(fp);
	return true;
}

bool BTFIAB::Init(const std::string& materialDirectoryPath)
{
	std::string allMaterialInfo;
	std::string allMaterialInfoPath = materialDirectoryPath + "/all_materialInfo.txt";
	bool avoidParFile = false;
	try
	{
		allMaterialInfo = FileIO::LoadFileAsString(allMaterialInfoPath);
		avoidParFile = true;
	}
	catch (std::ifstream::failure e)
	{
		UNIENGINE_LOG("")
	}
	if (!avoidParFile)
	{
		UNIENGINE_ERROR("Failed to load BTF material");
		return false;
	}
#pragma region Line 82 from ibtfbase.cpp
	m_materialOrder = 0;
	m_nColor = 0;
	// initial size of arrays
	m_maxPdf1D = 10000;
	m_maxVectorColor = 10000;
	m_maxIndexSlices = 10000;
	m_maxPdf2D = 10000;
	m_maxPdf2DLuminanceColor = 10000;
	m_maxPdf3D = 10000;
	m_maxPdf4D = 10000;
	m_maxPdf34D = 1000;

	// How the beta is discretized, either uniformly in degrees
	// or uniformly in cosinus of angle
	m_useCosBeta = true;
#pragma endregion
#pragma region Tilemap
	//Since tilemap is not used, the code here is not implemented.
#pragma endregion
#pragma region Scale info
	m_mPostScale = 1.0f;
	//Since no material contains the scale.txt is not used, the code here is not implemented.
#pragma endregion
#pragma region material info

	FILE* fp;
	if ((fp = fopen(allMaterialInfoPath.c_str(), "r")) == NULL) {
		UNIENGINE_ERROR("Failed to load BTF material");
		return false;
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
		return false;
	}
	// Here we need to read this information about original data
	int ncolour, nview, nillu, tileSize;
	if (fscanf(fp, "%d\n%d\n%d\n%d\n", &ncolour, &nview, &nillu, &tileSize) != 4) {
		fclose(fp);
		printf("File is corrupted for reading basic parameters about orig database\n");
		return false;
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
		return false;
	}
	m_useCosBeta = useCosBetaFlag ? true : false;
	m_lengthOfSlice = stepsPerBeta;
	assert((m_lengthOfSlice % 2) == 1);
	m_slicesPerHemisphere = stepsPerAlpha;
	assert((m_slicesPerHemisphere % 2) == 1);
	m_slicePerTheta = stepsPerTheta;
	assert(m_slicePerTheta >= 2);
	m_slicePerPhi = stepsPerPhi;
	assert(m_slicePerPhi >= 1);
#pragma endregion
#pragma region Create shared variables
	std::vector<float> betaAngles;
	// we always must have odd number of quantization steps per 180 degrees
	assert((this->m_lengthOfSlice % 2) == 1);
	if (m_useCosBeta) {
		printf("We use cos beta quantization with these values:\n");
		betaAngles.resize(m_lengthOfSlice);
		for (int i = 0; i < m_lengthOfSlice; i++) {
			float sinBeta = -1.0f + 2.0f * i / (m_lengthOfSlice - 1);
			if (sinBeta > 1.0f)
				sinBeta = 1.0f;
			// in degrees
			betaAngles[i] = 180.f / glm::pi<float>() * asin(sinBeta);
			printf("%3.2f ", betaAngles[i]);
		}
		printf("\n");
		betaAngles[0] = -90.f;
		betaAngles[(m_lengthOfSlice - 1) / 2] = 0.f;
		betaAngles[m_lengthOfSlice - 1] = 90.f;
	}
	else {
		float stepBeta = 0.f;
		// uniform quantization in angle
		printf("We use uniform angle quantization with these values:\n");
		stepBeta = 180.f / (m_lengthOfSlice - 1);
		betaAngles.resize(m_lengthOfSlice);
		for (int i = 0; i < m_lengthOfSlice; i++) {
			betaAngles[i] = i * stepBeta - 90.f;
			printf("%3.2f ", betaAngles[i]);
		}
		printf("\n");
		betaAngles[(m_lengthOfSlice - 1) / 2] = 0.f;
		betaAngles[m_lengthOfSlice - 1] = 90.0f;
	}

	// Here we set alpha
	m_stepAlpha = 180.f / (float)(m_slicesPerHemisphere - 1);

	m_tcTemplate = SharedCoordinates(m_useCosBeta, m_lengthOfSlice, betaAngles);

	m_tcTemplate.m_useCosBeta = m_useCosBeta;
	m_tcTemplate.m_lengthOfSlice = m_lengthOfSlice;
	// Setting alpha
	m_tcTemplate.m_stepAlpha = m_stepAlpha;
	m_tcTemplate.m_slicesPerHemi = m_slicesPerHemisphere;
	// Setting theta
	m_tcTemplate.m_slicesPerTheta = m_slicePerTheta;
	// Setting phi
	m_tcTemplate.m_slicesPerPhi = m_slicePerPhi;
	// use a specific flag when processing data from code BTF
	m_tcTemplate.m_codeBtfFlag = tmp12;
#pragma endregion
#pragma region Current settings
	// Here we need to read this information about current material setting
	// where are the starting points for the next search, possibly
	int fPDF1, fAB, fIAB, fPDF2, fPDF2L, fPDF2AB, fPDF3, fPDF34, fPDF4, fRESERVE;
	if (fscanf(fp, "%d %d %d %d %d %d %d %d %d %d\n",
		&fPDF1, &fAB, &fIAB, &fPDF2, &fPDF2L,
		&fPDF2AB, &fPDF3, &fPDF34, &fPDF4, &fRESERVE) != 10) {
		fclose(fp);
		UNIENGINE_ERROR("File is corrupted for reading starting search settings\n");
		return false;
	}
	// Here we need to save this information about current material setting
	int lsPDF1, lsAB, lsIAB, lsPDF2, lsPDF2L, lsPDF2AB, lsPDF3, lsPDF34, lsPDF4, lsRESERVE;
	if (fscanf(fp, "%d %d %d %d %d %d %d %d %d %d\n",
		&lsPDF1, &lsAB, &lsIAB, &lsPDF2, &lsPDF2L,
		&lsPDF2AB, &lsPDF3, &lsPDF34, &lsPDF4, &lsRESERVE) != 10) {
		fclose(fp);
		UNIENGINE_ERROR("File is corrupted for reading starting search points\n");
		return false;
	}

	int metric;
	float baseEps, rPDF1, epsAB, epsIAB, rPDF2, rPDF2L, epsPDF2AB, rPDF3, rPDF34, rPDF4, rPDF4b;
	if (fscanf(fp, "%d %f %f %f %f %f %f %f %f %f %f %f\n", &metric, &baseEps,
		&rPDF1, &epsAB, &epsIAB, &rPDF2, &rPDF2L,
		&epsPDF2AB, &rPDF3, &rPDF34, &rPDF4, &rPDF4b) != 12) {
		fclose(fp);
		UNIENGINE_ERROR("File is corrupted for reading epsilon search settings\n");
		return false;
	}
#pragma endregion

	// !!!!!! If we have only one database for all materials or
	// we share some databases except PDF6 for all materials
	m_use34ViewRepresentation = flagUse34DviewRep;
	m_usePdf2CompactRep = flagUsePDF2compactRep;

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

#pragma region Allocate arrays
	if (!m_allMaterialsInOneDatabase && loadMaterials != 1)
	{
		UNIENGINE_ERROR("Database for multiple materials are not supported!");
		return false;
	}
	//Here we only allow single material, so the array representations in original MLVQ lib are not implemented.
#pragma endregion
#pragma region Read paths
	std::string materialName;
	std::string inputPath;
	std::string outputPath;
	std::string tempPath;
	float hdrValue = 1.0f;
	int ro, co, pr, pc;
	char l1[1000], l2[1000], l3[1000], l4[1000];
	int hdrFlag = 0;
	if (fscanf(fp, "%s %s %s %s %d %d %d %d %f\n", l1, l2, l3, l4, &ro, &co, &pr, &pc, &hdrValue) == 9)
	{
		// Here we need to allocate the arrays for names
		materialName = std::string(l1);
		inputPath = std::string(l2);
		outputPath = std::string(l3);
		tempPath = std::string(l4);


		if (fabs(hdrValue - 1.0f) < 1e-6 ||
			fabs(hdrValue) < 1e-6) {
			hdrFlag = 0;
			hdrValue = 1.0f;
		}
		else {
			hdrFlag = 1;
		}
		m_tcTemplate.m_hdrFlag = hdrFlag;
		m_hdr = hdrFlag;
		m_hdrValue = hdrValue;
	}
	fclose(fp);
#pragma endregion

#pragma region Load material
	// Note that nrows and ncols are not set during loading !
	std::string fileName = materialDirectoryPath + "/" + materialName + "_materialInfo.txt";
	// Now creating PDF6 for each material using common database
	printf("Loading materials for common DBF1,DBF2,DBF3,DBF4,AB,IAB database\n");
	if ((fp = fopen(fileName.c_str(), "r")) == NULL) {
		UNIENGINE_ERROR("Cannot open file" + fileName);
		return true;
	}
	char nameM[200];
	if (fscanf(fp, "%s %s %s %s %d %d %d %d %f\n", &(nameM[0]),
		l1, l2, l3, &ro, &co, &pr, &pc, &hdrValue)
		!= 9) {
		UNIENGINE_ERROR("ERROR:Reading the information about material failed\n");
		printf("Exiting\n");
		fclose(fp); exit(-1);
	}
	inputPath = std::string(l1);
	outputPath = std::string(l2);
	tempPath = std::string(l3);
	fclose(fp);
	if ((fabs(hdrValue - 1.0f) < 1e-6) ||
		(fabs(hdrValue) < 1e-6)) {
		hdrFlag = 0;
		hdrValue = 1.0f;
	}
	else {
		hdrFlag = 1;
	}

	if (strcmp(nameM, materialName.c_str()) != 0) {
		UNIENGINE_ERROR("Some problem material name in file=" + std::string(nameM) + " other name=" + materialName + "\n", );
		return false;
	}
	// Now we can create the database, PDF6 is allocated
	// with right values
	m_tcTemplate.m_hdrFlag = hdrFlag;
	m_hdr = hdrFlag;
	m_hdrValue = hdrValue;
	
	auto& ab = m_pdf6.m_pdf4.m_pdf3.m_pdf2.m_iab.m_ab;
	auto& iab = m_pdf6.m_pdf4.m_pdf3.m_pdf2.m_iab;
	auto& pdf1 = m_pdf6.m_pdf4.m_pdf3.m_pdf2.m_pdf1;
	auto& pdf2 = m_pdf6.m_pdf4.m_pdf3.m_pdf2;
	auto& pdf3 = m_pdf6.m_pdf4.m_pdf3;
	auto& pdf4 = m_pdf6.m_pdf4;
	pdf1.Init(m_maxPdf1D, m_lengthOfSlice);

	ab.Init(m_maxVectorColor);
	iab.Init(m_maxIndexSlices, m_lengthOfSlice);
	pdf2.Init(m_maxPdf2D, m_slicesPerHemisphere);
	m_tcTemplate.m_slicesPerHemi = pdf2.m_numOfSlicesPerHemisphere;

	pdf3.Init(m_maxPdf3D, m_slicePerTheta);
	m_tcTemplate.m_slicesPerTheta = pdf3.m_slicesPerTheta;
	m_tcTemplate.m_stepTheta = pdf3.m_stepTheta;

	pdf4.Init(m_maxPdf4D, m_slicePerPhi);
	assert(m_tcTemplate.m_slicesPerPhi == pdf4.m_slicesPerPhi);
	m_tcTemplate.m_stepPhi = pdf4.m_stepPhi;

	m_pdf6.Init(pr, pc, ro, co, m_nColor);
	
#pragma region Load Data
	std::vector<int> intData;
	std::vector<float> floatData;
	std::string prefix = materialDirectoryPath + "/" + materialName;
	int minIntVal, maxIntVal;
	float minFloatVal, maxFloatVal;
	
	ParseIntData(prefix + "_PDF6Dslices.txt", m_pdf6.m_numOfRows, m_pdf6.m_numOfCols, minIntVal, maxIntVal, intData);
	ParseFloatData(prefix + "_PDF6Dscale.txt", m_pdf6.m_numOfRows, m_pdf6.m_numOfCols, minFloatVal, maxFloatVal, floatData);
	m_pdf6.m_pdf6DSlicesBuffer.Upload(intData);
	m_pdf6.m_pdf6DScaleBuffer.Upload(floatData);
	m_pdf6.m_pdf6DSlices = reinterpret_cast<int*>(m_pdf6.m_pdf6DSlicesBuffer.DevicePointer());
	m_pdf6.m_pdf6DScale = reinterpret_cast<float*>(m_pdf6.m_pdf6DScaleBuffer.DevicePointer());

	prefix = materialDirectoryPath + "/" + "all";

	ParseFloatData(prefix + "_PDF1Dslice.txt", pdf1.m_numOfPdf1D, pdf1.m_lengthOfSlice, minFloatVal, maxFloatVal, floatData);
	pdf1.m_pdf1DBasisBuffer.Upload(floatData);
	pdf1.m_pdf1DBasis = reinterpret_cast<float*>(pdf1.m_pdf1DBasisBuffer.DevicePointer());

	ParseFloatData(prefix + "_colors.txt", ab.m_numOfColors, ab.m_numOfChannels, minFloatVal, maxFloatVal, floatData);
	ab.m_vectorColorBasisBuffer.Upload(floatData);
	ab.m_vectorColorBasis = reinterpret_cast<float*>(ab.m_vectorColorBasisBuffer.DevicePointer());

	ParseIntData(prefix + "_indexAB.txt", iab.m_numOfIndexSlices, iab.m_lengthOfSlice, minIntVal, maxIntVal, intData);
	iab.m_indexAbBasisBuffer.Upload(intData);
	iab.m_indexAbBasis = reinterpret_cast<int*>(iab.m_indexAbBasisBuffer.DevicePointer());

	ParseIntData(prefix + "_PDF2Dcolours.txt", pdf2.m_color.m_numOfPdf2D, pdf2.m_color.m_slicesPerHemisphere, minIntVal, maxIntVal, intData);
	pdf2.m_color.m_pdf2DColorsBuffer.Upload(intData);
	pdf2.m_color.m_pdf2DColors = reinterpret_cast<int*>(pdf2.m_color.m_pdf2DColorsBuffer.DevicePointer());
	ParseIntData(prefix + "_PDF2Dslices.txt", pdf2.m_luminance.m_numOfPdf2D, pdf2.m_luminance.m_slicesPerHemisphere, minIntVal, maxIntVal, intData);
	pdf2.m_luminance.m_pdf2DSlicesBuffer.Upload(intData);
	pdf2.m_luminance.m_pdf2DSlices = reinterpret_cast<int*>(pdf2.m_luminance.m_pdf2DSlicesBuffer.DevicePointer());
	ParseFloatData(prefix + "_PDF2Dscale.txt", pdf2.m_luminance.m_numOfPdf2D, pdf2.m_luminance.m_slicesPerHemisphere, minFloatVal, maxFloatVal, floatData);
	pdf2.m_luminance.m_pdf2DScalesBuffer.Upload(floatData);
	pdf2.m_luminance.m_pdf2DScales = reinterpret_cast<float*>(pdf2.m_luminance.m_pdf2DScalesBuffer.DevicePointer());
	ParseIntData(prefix + "_PDF2Dindices.txt", pdf2.m_numOfPdf2D, pdf2.m_lengthOfSlice, minIntVal, maxIntVal, intData);
	pdf2.m_indexLuminanceColorBuffer.Upload(intData);
	pdf2.m_indexLuminanceColor = reinterpret_cast<int*>(pdf2.m_indexLuminanceColorBuffer.DevicePointer());
	
	ParseFloatData(prefix + "_PDF3Dscale.txt", pdf3.m_numOfPdf3D, pdf3.m_slicesPerTheta, minFloatVal, maxFloatVal, floatData);
	pdf3.m_pdf3DScalesBuffer.Upload(floatData);
	pdf3.m_pdf3DScales = reinterpret_cast<float*>(pdf3.m_pdf3DScalesBuffer.DevicePointer());
	ParseIntData(prefix + "_PDF3Dslices.txt", pdf3.m_numOfPdf3D, pdf3.m_slicesPerTheta, minIntVal, maxIntVal, intData);
	pdf3.m_pdf3DSlicesBuffer.Upload(intData);
	pdf3.m_pdf3DSlices = reinterpret_cast<int*>(pdf3.m_pdf3DSlicesBuffer.DevicePointer());

	ParseFloatData(prefix + "_PDF4Dscale.txt", pdf4.m_numOfPdf4D, pdf4.m_slicesPerPhi, minFloatVal, maxFloatVal, floatData);
	pdf4.m_pdf4DScalesBuffer.Upload(floatData);
	pdf4.m_pdf4DScales = reinterpret_cast<float*>(pdf4.m_pdf4DScalesBuffer.DevicePointer());
	ParseIntData(prefix + "_PDF4Dslices.txt", pdf4.m_numOfPdf4D, pdf4.m_slicesPerPhi, minIntVal, maxIntVal, intData);
	pdf4.m_pdf4DSlicesBuffer.Upload(intData);
	pdf4.m_pdf4DSlices = reinterpret_cast<int*>(pdf4.m_pdf4DSlicesBuffer.DevicePointer());
	UNIENGINE_LOG("The database was read successfully.");
#pragma endregion
	return true; // OK - database loaded, or at least partially

#pragma endregion
}
