/*!**************************************************************************
\file    PDF2D.cpp
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The main file for the:  2D PDF database
******************************************************************************/
#include <cassert>
#include <CIELab.hpp>
#include <PDF1D.hpp>
#include <IndexAB.hpp>
#include <PDF2D.hpp>

//#########################################################################
//######## CPDF2D #################################################
//#########################################################################
CPDF2D::CPDF2D(int maxPdf2D, int slicesPerHemisphere,
	CPDF1D* pdf1, CIndexAB* iab)
{
	assert(maxPdf2D > 0);
	assert(slicesPerHemisphere > 0);
	assert(pdf1);
	assert(iab);

	this->m_maxPdf2D = maxPdf2D;
	this->m_numOfPdf2D = 0;
	this->m_numOfSlicesPerHemisphere = slicesPerHemisphere;
	this->m_size2D = slicesPerHemisphere * pdf1->GetSliceLength();
	this->m_pdf1 = pdf1;
	this->m_iab = iab;
}

CPDF2D::~CPDF2D()
{
	// in abstract class nothing to do
}//--- CPDF2D ---------------------------------------------------------

