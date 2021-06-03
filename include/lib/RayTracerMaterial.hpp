#pragma once
#include <memory>
#include <UniEngine-pch.hpp>
#include <raymlvq_export.h>
#include <Entity.hpp>
#include <Texture2D.hpp>

using namespace UniEngine;
namespace RayMLVQ {
	class RAYMLVQ_API RayTracerMaterial :
		public PrivateComponentBase
	{
	public:
		float m_diffuseIntensity = 0;
		std::shared_ptr<Texture2D> m_albedoTexture;
		std::shared_ptr<Texture2D> m_normalTexture;
		void OnGui() override;
	};
}
