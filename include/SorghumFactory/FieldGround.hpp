#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace PlantArchitect {
	struct NoiseDescriptor
	{
		float m_noiseScale = 5.0f;
		float m_noiseIntensity = 0.025f;
		float m_heightMin = -10;
		float m_heightMax = 10;
	};
	class SORGHUM_FACTORY_API FieldGround : public IAsset {
	public:
		glm::vec2 m_scale;
		glm::ivec2 m_size;
		float m_rowWidth;
		float m_alleyDepth;

		std::vector<NoiseDescriptor> m_noiseDescriptors;
		void OnCreate() override;
		Entity GenerateMesh(float overrideDepth = -1.0f);
		void OnInspect() override;
		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
	};
}