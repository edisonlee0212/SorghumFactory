#include <RayTracedRenderer.hpp>
#include <EditorManager.hpp>
#include <MeshRenderer.hpp>

using namespace RayTracerFacility;
using namespace UniEngine;

const char *MLVQMaterials[]{
        "fabrics01",
        "alu",
        "corduroy",
        "wool",
        "wallpaper",
        "impalla",
        "pulli",
        "proposte"
};

void RayTracedRenderer::OnGui() {
    ImGui::DragFloat("Metallic##RayTracedRenderer", &m_metallic, 0.01f, 0.0f, 1.0f);
    ImGui::DragFloat("Roughness##RayTracedRenderer", &m_roughness, 0.01f, 0.0f, 1.0f);
    ImGui::DragFloat("Transparency##RayTracedRenderer", &m_transparency, 0.01f, 0.0f, 1.0f);
    ImGui::DragFloat("Diffuse intensity##RayTracedRenderer", &m_diffuseIntensity, 0.01f, 0.0f, 100.0f);
    ImGui::ColorEdit3("Surface Color##RayTracedRenderer", &m_surfaceColor.x);
    EditorManager::DragAndDrop<Mesh>(m_mesh, "Mesh");
    if (m_mesh.Get<Mesh>()) {
        if (ImGui::TreeNode("Mesh##MeshRenderer")) {
          m_mesh.Get<Mesh>()->OnGui();
            ImGui::TreePop();
        }
    }
    ImGui::Text("Material: ");
    ImGui::SameLine();
    ImGui::Checkbox("MLVQ##RayTracedRenderer", &m_enableMLVQ);
    if (m_enableMLVQ) {
        ImGui::Combo("MLVQ Material", &m_mlvqMaterialIndex, MLVQMaterials, IM_ARRAYSIZE(MLVQMaterials));
    } else {
        if (ImGui::TreeNode("Textures##RayTracerMaterial")) {
            EditorManager::DragAndDrop<Texture2D>(m_albedoTexture, "Albedo");


            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Text("Normal:");
            ImGui::SameLine();
            EditorManager::DragAndDrop<Texture2D>(m_normalTexture, "Normal");

            ImGui::TreePop();
        }
    }
}

void RayTracedRenderer::SyncWithMeshRenderer() {
    Entity owner = GetOwner();
    if (owner.HasPrivateComponent<MeshRenderer>()) {
        auto mmr = owner.GetOrSetPrivateComponent<MeshRenderer>().lock();
        m_mesh = mmr->m_mesh;
        auto mat = mmr->m_material.Get<Material>();
        m_roughness = mat->m_roughness;
        m_metallic = mat->m_metallic;
        m_surfaceColor = mat->m_albedoColor;
    }
}
void RayTracedRenderer::Clone(const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<RayTracedRenderer>(target);
}
