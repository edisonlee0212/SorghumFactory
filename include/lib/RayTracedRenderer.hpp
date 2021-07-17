#pragma once
#include <UniEngine-pch.hpp>
#include <memory>
#include <ray_tracer_facility_export.h>

#include <Entity.hpp>
#include <Mesh.hpp>
#include <Texture2D.hpp>

using namespace UniEngine;
namespace RayTracerFacility {
class RAY_TRACER_FACILITY_API RayTracedRenderer : public IPrivateComponent {
public:
  float m_diffuseIntensity = 0;
  float m_transparency = 1.0f;
  float m_metallic = 0.3f;
  float m_roughness = 0.3f;
  glm::vec3 m_surfaceColor = glm::vec3(1.0f);
  std::shared_ptr<Mesh> m_mesh;
  std::shared_ptr<Texture2D> m_albedoTexture;
  std::shared_ptr<Texture2D> m_normalTexture;
  bool m_enableMLVQ = false;
  int m_mlvqMaterialIndex = 0;
  void OnGui() override;
  void SyncWithMeshRenderer();
};
} // namespace RayTracerFacility
