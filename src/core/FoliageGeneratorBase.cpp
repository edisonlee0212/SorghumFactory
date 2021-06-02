#include <FoliageGeneratorBase.hpp>
#include <PlantManager.hpp>
using namespace PlantFactory;
std::shared_ptr<Texture2D> DefaultFoliageGenerator::_LeafSurfaceTex = nullptr;


void DefaultFoliageGenerator::GenerateLeaves(Entity& internode, glm::mat4& treeTransform,
                                                           std::vector<glm::mat4>& leafTransforms, bool isLeft)
{
}

DefaultFoliageGenerator::DefaultFoliageGenerator()
{
	_DefaultFoliageInfo = DefaultFoliageInfo();
	_Archetype = EntityManager::CreateEntityArchetype("Pine Foliage", DefaultFoliageInfo());

	_LeafMaterial = std::make_shared<Material>();
	_LeafMaterial->m_shininess = 32.0f;
	_LeafMaterial->SetProgram(Default::GLPrograms::StandardInstancedProgram);
	_LeafMaterial->m_alphaDiscardEnabled = true;
	_LeafMaterial->m_alphaDiscardOffset = 0.1f;
	_LeafMaterial->m_cullingMode = MaterialCullingMode::Off;
	if (!_LeafSurfaceTex) _LeafSurfaceTex = ResourceManager::LoadTexture(false, FileIO::GetAssetFolderPath() + "Textures/Leaf/Pine/level0.png");
	//_LeafMaterial->SetTexture(_LeafSurfaceTex);
	_LeafMaterial->m_albedoColor = glm::normalize(glm::vec3(60.0f / 256.0f, 140.0f / 256.0f, 0.0f));
	_LeafMaterial->m_metallic = 0.0f;
	_LeafMaterial->m_roughness = 0.3f;
	_LeafMaterial->m_ambientOcclusion = glm::linearRand(0.4f, 0.8f);
}

void DefaultFoliageGenerator::Generate()
{
	const auto tree = GetOwner();
	GlobalTransform treeTransform = EntityManager::GetComponentData<GlobalTransform>(tree);
	Entity foliageEntity;
	bool found = false;
	EntityManager::ForEachChild(tree, [&found, &foliageEntity](Entity child)
		{
			if (child.HasComponentData<DefaultFoliageInfo>())
			{
				found = true;
				foliageEntity = child;
			}
		}
	);
	if (!found)
	{
		foliageEntity = EntityManager::CreateEntity(_Archetype, "Foliage");
		EntityManager::SetParent(foliageEntity, tree);
		auto particleSys = std::make_unique<Particles>();
		particleSys->m_material = _LeafMaterial;
		particleSys->m_mesh = Default::Primitives::Quad;
		particleSys->m_forwardRendering = false;
		Transform transform;
		transform.m_value = glm::translate(glm::vec3(0.0f)) * glm::scale(glm::vec3(1.0f));
		foliageEntity.SetPrivateComponent(std::move(particleSys));
		foliageEntity.SetComponentData(transform);
		foliageEntity.SetComponentData(_DefaultFoliageInfo);
	}
	auto& particleSys = foliageEntity.GetPrivateComponent<Particles>();
	particleSys->m_matrices.clear();
	GenerateLeaves(EntityManager::GetChildren(tree)[0], treeTransform.m_value, particleSys->m_matrices, true);
}

void DefaultFoliageGenerator::OnGui()
{
	if (ImGui::Button("Regenerate")) Generate();
	ImGui::DragFloat2("Leaf Size XY", static_cast<float*>(static_cast<void*>(&_DefaultFoliageInfo.LeafSize)), 0.01f);
	ImGui::DragFloat("LeafIlluminationLimit", &_DefaultFoliageInfo.LeafIlluminationLimit, 0.01f);
	ImGui::DragFloat("LeafInhibitorFactor", &_DefaultFoliageInfo.LeafInhibitorFactor, 0.01f);
	ImGui::Checkbox("IsBothSide", &_DefaultFoliageInfo.IsBothSide);
	ImGui::DragInt("SideLeafAmount", &_DefaultFoliageInfo.SideLeafAmount, 0.01f);
	ImGui::DragFloat("StartBendingAngle", &_DefaultFoliageInfo.StartBendingAngle, 0.01f);
	ImGui::DragFloat("BendingAngleIncrement", &_DefaultFoliageInfo.BendingAngleIncrement, 0.01f);
	ImGui::DragFloat("LeafPhotoTropism", &_DefaultFoliageInfo.LeafPhotoTropism, 0.01f);
	ImGui::DragFloat("LeafGravitropism", &_DefaultFoliageInfo.LeafGravitropism, 0.01f);
	ImGui::DragFloat("LeafDistance", &_DefaultFoliageInfo.LeafDistance, 0.01f);
}
