#pragma once
#include <sorghum_factory_export.h>

#include <SorghumProceduralDescriptor.hpp>
using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API DepthCamera : public IPrivateComponent, public RenderTarget{
  static std::shared_ptr<OpenGLUtils::GLProgram> m_depthTransferProgram;
  static std::shared_ptr<OpenGLUtils::GLVAO> m_depthTransferVAO;
public:
  bool m_useCameraResolution = true;
  int m_resX = 1;
  int m_resY = 1;
  void Update() override;
  void OnCreate() override;
  std::shared_ptr<Texture2D> m_colorTexture;
  void OnInspect() override;
  void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
  void ExportAsYaml(const std::filesystem::path& exportPath);
};
} // namespace SorghumFactory