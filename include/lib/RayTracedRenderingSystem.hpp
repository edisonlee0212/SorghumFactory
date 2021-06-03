#pragma once
#include <raymlvq_export.h>
#include <CUDAModule.hpp>
#include <UniEngine-pch.hpp>
#include <Application.hpp>

using namespace UniEngine;
namespace RayMLVQ
{
	class RAYMLVQ_API RayTracedRenderingSystem : public SystemBase
	{
		void OnGui();
	public:
		void UpdateDebugRenderOutputScene();
		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;
		void Update() override;
		void FixedUpdate() override;
		void LateUpdate() override;
#pragma region Rendering
		DebugRenderingProperties m_properties;
		float m_cameraFov = 60;
		bool m_rayTracerDebugRenderingEnabled = true;
		float m_lastX = 0;
		float m_lastY = 0;
		float m_lastScrollY = 0;
		bool m_startMouse = false;
		bool m_startScroll = false;
		bool m_rightMouseButtonHold = false;
		
		std::unique_ptr<OpenGLUtils::GLTexture2D> m_rayTracerTestOutput;
		glm::ivec2 m_rayTracerTestOutputSize = glm::ivec2(1024, 1024);
		bool m_rendered = false;
		std::shared_ptr<Cubemap> m_environmentalMap;
#pragma endregion
		
	};
}