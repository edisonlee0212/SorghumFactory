#pragma once
#include <memory>
#include <UniEngine-pch.hpp>
#include <raymlvq_export.h>
#include <Entity.hpp>
#include <Texture2D.hpp>
#include <Mesh.hpp>

using namespace UniEngine;
namespace RayMLVQ {
	class RAYMLVQ_API RayTracedRenderer :
		public PrivateComponentBase
	{
	public:
		float m_diffuseIntensity = 0;
		float m_transparency = 1.0f;
		float m_metallic = 0.3f;
		float m_roughness = 0.3f;
		glm::vec3 m_surfaceColor = glm::vec3(1.0f);
		std::shared_ptr<Mesh> m_mesh;
		std::shared_ptr<Texture2D> m_albedoTexture;
		std::shared_ptr<Texture2D> m_normalTexture;
		void OnGui() override;
		void SyncWithMeshRenderer();
	};
}
