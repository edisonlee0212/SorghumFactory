#include <RayTracedRenderer.hpp>
#include <EditorManager.hpp>
#include <MeshRenderer.hpp>
using namespace RayTracerFacility;
using namespace UniEngine;
void RayTracedRenderer::OnGui()
{
	ImGui::DragFloat("Metallic##RayTracedRenderer", &m_metallic, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("Roughness##RayTracedRenderer", &m_roughness, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("Transparency##RayTracedRenderer", &m_transparency, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("Diffuse intensity##RayTracedRenderer", &m_diffuseIntensity, 0.01f, 0.0f, 100.0f);
	ImGui::ColorEdit3("Surface Color##RayTracedRenderer", &m_surfaceColor.x);
	ImGui::Checkbox("MLVQ##RayTracedRenderer", &m_enableMLVQ);
	EditorManager::DragAndDrop(m_mesh);
	if (m_mesh)
	{
		if (ImGui::TreeNode("Mesh##MeshRenderer"))
		{
			m_mesh->OnGui();
			ImGui::TreePop();
		}
	}
	
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

void RayTracedRenderer::SyncWithMeshRenderer()
{
	Entity owner = GetOwner();
	if(owner.HasPrivateComponent<MeshRenderer>())
	{
		auto& mmr = owner.GetPrivateComponent<MeshRenderer>();
		m_mesh = mmr->m_mesh;
		m_roughness = mmr->m_material->m_roughness;
		m_metallic = mmr->m_material->m_metallic;
		m_surfaceColor = mmr->m_material->m_albedoColor;
	}
}
