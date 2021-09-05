//
// Created by lllll on 9/5/2021.
//

#include "DepthCamera.hpp"

using namespace SorghumFactory;

std::shared_ptr<OpenGLUtils::GLProgram> DepthCamera::m_depthTransferProgram;
std::shared_ptr<OpenGLUtils::GLVAO> DepthCamera::m_depthTransferVAO;

void DepthCamera::Clone(const std::shared_ptr<IPrivateComponent> &target) {}
void DepthCamera::OnInspect() {}
void DepthCamera::Update() {
  if (!GetOwner().HasPrivateComponent<Camera>())
    return;
  auto cameraComponent = GetOwner().GetOrSetPrivateComponent<Camera>().lock();
  // 1. Resize to camera's resolution
  auto resolution = cameraComponent->GetResolution();
  if (m_resolutionX != resolution.x || m_resolutionY != resolution.y) {
    m_resolutionX = resolution.x;
    m_resolutionY = resolution.y;
    m_colorTexture->Texture()->ReSize(0, GL_RGB16F, GL_RGB, GL_FLOAT, 0,
                                      m_resolutionX, m_resolutionY);
  }
  // 2. Render to depth texture
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  m_depthTransferVAO->Bind();

  m_depthTransferProgram->Bind();

  AttachTexture(m_colorTexture->Texture().get(), GL_COLOR_ATTACHMENT0);
  Bind();
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  cameraComponent->GetDepthStencil()->Texture()->Bind(1);
  m_depthTransferProgram->SetInt("depthStencil", 0);

  glDrawArrays(GL_TRIANGLES, 0, 6);
}
void DepthCamera::OnCreate() {
  if (!m_depthTransferProgram) {
    auto fragShaderCode =
        std::string("#version 450 core\n") +
        FileUtils::LoadFileAsString(std::filesystem::path("../Resources") /
                                    "Shaders/Fragment/DepthCopy.frag");
    auto fragShader = AssetManager::CreateAsset<OpenGLUtils::GLShader>();
    fragShader->Set(OpenGLUtils::ShaderType::Fragment, fragShaderCode);
    m_depthTransferProgram =
        AssetManager::CreateAsset<OpenGLUtils::GLProgram>();
    m_depthTransferProgram->Link(
        DefaultResources::GLShaders::TexturePassThrough, fragShader);
  }

  if(!m_depthTransferVAO){
    float quadVertices[] = {
        // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        // positions   // texCoords
        -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  1.0f, 1.0f};

    m_depthTransferVAO = std::make_shared<OpenGLUtils::GLVAO>();
    m_depthTransferVAO->SetData(sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    m_depthTransferVAO->EnableAttributeArray(0);
    m_depthTransferVAO->SetAttributePointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    m_depthTransferVAO->EnableAttributeArray(1);
    m_depthTransferVAO->SetAttributePointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
  }

  m_resolutionX = 1;
  m_resolutionY = 1;

  m_colorTexture = AssetManager::CreateAsset<Texture2D>();
  m_colorTexture->m_name = "CameraTexture";
  m_colorTexture->Texture() = std::make_shared<OpenGLUtils::GLTexture2D>(
      0, GL_RGB16F, m_resolutionX, m_resolutionY, false);
  m_colorTexture->Texture()->SetData(0, GL_RGB16F, GL_RGB, GL_FLOAT, 0);
  m_colorTexture->Texture()->SetInt(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  m_colorTexture->Texture()->SetInt(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  m_colorTexture->Texture()->SetInt(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  m_colorTexture->Texture()->SetInt(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  AttachTexture(m_colorTexture->Texture().get(), GL_COLOR_ATTACHMENT0);
}
