#include <RayTracerMaterial.hpp>
using namespace PlantFactory;
using namespace UniEngine;
void RayTracerMaterial::OnGui()
{
	ImGui::DragFloat("Diffuse intensity", &m_diffuseIntensity, 0.01f, 0.0f, 100.0f);
	if (ImGui::TreeNode("Textures##RayTracerMaterial")) {

		ImGui::Spacing();
		ImGui::Separator();
		ImGui::Text("Albedo:");
		ImGui::SameLine();
		EditorManager::DragAndDrop(m_albedoTexture);



		ImGui::Spacing();
		ImGui::Separator();
		ImGui::Text("Normal:");
		ImGui::SameLine();
		EditorManager::DragAndDrop(m_normalTexture);

		ImGui::TreePop();
	}
}
