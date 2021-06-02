#pragma once
using namespace UniEngine;
namespace PlantFactory {
	class SorghumParameters
	{
	public:
		void OnGui();
		void Serialize(const std::string& path) const;
		void Deserialize(const std::string& path);

		int m_leafCount = 8;
		
		float m_branchAngleMean = 20;
		float m_branchAngleVariance = 0;

		float m_rollAngleVariance = 1.0f;
		float m_rollAngleVarianceDistanceFactor = 1.0f;

		float m_internodeLength = 0.4f;
		float m_internodeLengthVariance = 0.1f;
		
		float m_leafGravityBending = 7.0f;
		float m_leafGravityBendingLevelFactor = -1.0f;
		float m_leafGravityBendingIncreaseFactor = 1.0f;

		float m_leafLengthBase = 4.0f;
	};
}