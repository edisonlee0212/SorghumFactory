#pragma once
#include <raymlvq_export.h>
#include <CUDAModule.hpp>
#include <UniEngine-pch.hpp>
#include <memory>
#include <Application.hpp>
#include <EditorManager.hpp>
#include <Cubemap.hpp>
#include <InputManager.hpp>
#include <WindowManager.hpp>
#include <EditorManager.hpp>
#include <ResourceManager.hpp>
#include <MeshRenderer.hpp>
#include <Cubemap.hpp>
using namespace UniEngine;
namespace RayMLVQ
{
	class RayTracerRenderWindow
	{
	public:
		std::string m_name;
		
		float m_cameraFov = 60;
		bool m_renderingEnabled = true;
		float m_lastX = 0;
		float m_lastY = 0;
		float m_lastScrollY = 0;
		bool m_startMouse = false;
		bool m_startScroll = false;
		bool m_rightMouseButtonHold = false;

		std::unique_ptr<OpenGLUtils::GLTexture2D> m_output;
		glm::ivec2 m_outputSize = glm::ivec2(1024, 1024);
		bool m_rendered = false;
		void Init(const std::string& name);
		void Resize();
		void OnGui();
	};

	class RAYMLVQ_API RayTracerManager
	{
	protected:
#pragma region Class related
		RayTracerManager() = default;
		RayTracerManager(RayTracerManager&&) = default;
		RayTracerManager(const RayTracerManager&) = default;
		RayTracerManager& operator=(RayTracerManager&&) = default;
		RayTracerManager& operator=(const RayTracerManager&) = default;
#pragma endregion
	public:
		std::shared_ptr<Cubemap> m_environmentalMap;
		
		DefaultRenderingProperties m_defaultRenderingProperties;
		RayTracerRenderWindow m_defaultWindow;

		RayMLVQRenderingProperties m_rayMLVQRenderingProperties;
		RayTracerRenderWindow m_rayMLVQWindow;

		void UpdateScene() const;
		
		static RayTracerManager& GetInstance();
		static void Init();
		static void Update();
		static void OnGui();
		static void End();
	};
}