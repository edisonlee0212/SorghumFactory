#pragma once
#include <sorghum_factory_export.h>

#include <SorghumParameters.hpp>
using namespace UniEngine;
namespace SorghumFactory {
class DepthCamera : public IPrivateComponent, public RenderTarget{
  static std::shared_ptr<OpenGLUtils::GLProgram> m_depthTransferProgram;
  static std::shared_ptr<OpenGLUtils::GLVAO> m_depthTransferVAO;
public:
  float m_min;
  float m_max;
  void Update() override;
  void OnCreate() override;
  std::shared_ptr<Texture2D> m_colorTexture;
  void OnInspect() override;
  void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
};
} // namespace SorghumFactory