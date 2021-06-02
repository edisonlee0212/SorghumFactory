#pragma once
using namespace UniEngine;
namespace PlantFactory {
	class RayTracerMaterial :
		public PrivateComponentBase
	{
	public:
		float m_diffuseIntensity = 0;
		std::shared_ptr<Texture2D> m_albedoTexture;
		std::shared_ptr<Texture2D> m_normalTexture;
		void OnGui() override;
	};
}
