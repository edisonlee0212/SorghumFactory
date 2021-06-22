#include <TreeManager.hpp>
#include <CubeVolume.hpp>
#include <Curve.hpp>
#include <RadialBoundingVolume.hpp>
#include <RayTracedRenderer.hpp>
#include <TreeLeaves.hpp>

using namespace RayTracerFacility;
using namespace PlantFactory;

void TreeManager::ExportChains(int parentOrder, Entity internode, rapidxml::xml_node<>* chains,
	rapidxml::xml_document<>* doc)
{
	auto order = internode.GetComponentData<InternodeInfo>().m_order;
	if (order != parentOrder)
	{
		WriteChain(order, internode, chains, doc);
	}
	EntityManager::ForEachChild(internode, [order, &chains, doc](Entity child)
		{
			ExportChains(order, child, chains, doc);
		}
	);
}

void TreeManager::WriteChain(int order, Entity internode, rapidxml::xml_node<>* chains,
	rapidxml::xml_document<>* doc)
{
	Entity walker = internode;
	rapidxml::xml_node<>* chain = doc->allocate_node(rapidxml::node_element, "Chain", "Node");
	chain->append_attribute(doc->allocate_attribute("gravelius", doc->allocate_string(std::to_string(order + 1).c_str())));
	chains->append_node(chain);
	std::vector<rapidxml::xml_node<>*> nodes;
	while (EntityManager::GetChildrenAmount(walker) != 0)
	{
		auto* node = doc->allocate_node(rapidxml::node_element, "Node");
		node->append_attribute(doc->allocate_attribute("id", doc->allocate_string(std::to_string(walker.m_index).c_str())));
		nodes.push_back(node);
		//chain->append_node(node);
		Entity temp;
		EntityManager::ForEachChild(walker, [&temp, order](Entity child)
			{
				if (child.GetComponentData<InternodeInfo>().m_order == order)
				{
					temp = child;
				}
			}
		);
		walker = temp;
	}
	if (nodes.empty()) return;
	for (int i = nodes.size() - 1; i >= 0; i--)
	{
		chain->append_node(nodes[i]);
	}
	if (EntityManager::GetParent(internode).HasComponentData<InternodeInfo>())
	{
		auto* node = doc->allocate_node(rapidxml::node_element, "Node");
		node->append_attribute(doc->allocate_attribute("id", doc->allocate_string(std::to_string(EntityManager::GetParent(internode).m_index).c_str())));
		chain->append_node(node);
	}
}

Entity TreeManager::GetRootInternode(const Entity& tree)
{
	auto retVal = Entity();
	if (!tree.HasComponentData<PlantInfo>() || tree.GetComponentData<PlantInfo>().m_plantType != PlantType::GeneralTree) return retVal;
	EntityManager::ForEachChild(tree, [&](Entity child)
		{
			if (child.HasComponentData<InternodeInfo>()) retVal = child;
		}
	);
	return retVal;
}

Entity TreeManager::GetLeaves(const Entity& tree)
{
	auto retVal = Entity();
	if (!tree.HasComponentData<PlantInfo>() || tree.GetComponentData<PlantInfo>().m_plantType != PlantType::GeneralTree) return retVal;
	EntityManager::ForEachChild(tree, [&](Entity child)
		{
			if (child.HasComponentData<TreeLeavesTag>()) retVal = child;
		}
	);
	if (!retVal.IsValid())
	{
		const auto leaves = EntityManager::CreateEntity(GetInstance().m_leavesArchetype, "Leaves");
		EntityManager::SetParent(leaves, tree);
		leaves.SetPrivateComponent(std::make_unique<TreeLeaves>());
		leaves.SetPrivateComponent(std::make_unique<MeshRenderer>());
		leaves.SetPrivateComponent(std::make_unique<RayTracedRenderer>());
		auto& meshRenderer = leaves.GetPrivateComponent<MeshRenderer>();
		auto& rayTracerMaterial = leaves.GetPrivateComponent<RayTracedRenderer>();
		meshRenderer->m_material = ResourceManager::LoadMaterial(true, DefaultResources::GLPrograms::StandardProgram);
		meshRenderer->m_material->m_name = "Leaves mat";
		meshRenderer->m_material->m_roughness = 0.0f;
		meshRenderer->m_material->m_metallic = 0.7f;
		meshRenderer->m_material->m_albedoColor = glm::vec3(0.0f, 1.0f, 0.0f);
		meshRenderer->m_mesh = std::make_shared<Mesh>();
		rayTracerMaterial->SyncWithMeshRenderer();
	}
	return retVal;
}

Entity TreeManager::GetRbv(const Entity& tree)
{
	auto retVal = Entity();
	if (!tree.HasComponentData<PlantInfo>() || tree.GetComponentData<PlantInfo>().m_plantType != PlantType::GeneralTree) return retVal;
	EntityManager::ForEachChild(tree, [&](Entity child)
		{
			if (child.HasComponentData<RbvTag>()) retVal = child;
		}
	);
	if (!retVal.IsValid())
	{
		const auto rbv = EntityManager::CreateEntity(GetInstance().m_rbvArchetype, "RBV");
		EntityManager::SetParent(rbv, tree);
		rbv.SetPrivateComponent(std::make_unique<RadialBoundingVolume>());
	}
	return retVal;
}

void TreeManager::UpdateBranchCylinder(const bool& displayThickness, const float& width)
{
	auto& plantManager = PlantManager::GetInstance();
	EntityManager::ForEach<GlobalTransform, BranchCylinder, BranchCylinderWidth, InternodeGrowth, InternodeInfo>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [displayThickness, width]
	(int i, Entity entity, GlobalTransform& ltw, BranchCylinder& c, BranchCylinderWidth& branchCylinderWidth, InternodeGrowth& internodeGrowth, InternodeInfo& internodeInfo)
		{
			glm::vec3 scale;
			glm::quat rotation;
			glm::vec3 translation;
			glm::vec3 skew;
			glm::vec4 perspective;
			glm::decompose(ltw.m_value, scale, rotation, translation, skew, perspective);
			const glm::vec3 parentTranslation = EntityManager::GetParent(entity).GetComponentData<GlobalTransform>().GetPosition();
			const auto direction = glm::normalize(parentTranslation - translation);
			rotation = glm::quatLookAt(direction, glm::vec3(direction.y, direction.z, direction.x));
			rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
			const glm::mat4 rotationTransform = glm::mat4_cast(rotation);

			if (EntityManager::HasPrivateComponent<TreeData>(EntityManager::GetParent(entity))) {
				c.m_value = glm::translate((translation + parentTranslation) / 2.0f) * rotationTransform * glm::scale(glm::vec3(0.0f));
				branchCylinderWidth.m_value = 0;
			}
			else {
				branchCylinderWidth.m_value = displayThickness ? internodeGrowth.m_thickness : width;
				c.m_value = glm::translate((translation + parentTranslation) / 2.0f) * rotationTransform * glm::scale(glm::vec3(branchCylinderWidth.m_value, glm::distance(translation, parentTranslation) / 2.0f, displayThickness ? internodeGrowth.m_thickness : width));
			}
		}, false
		);
}

void TreeManager::UpdateBranchPointer(const float& length, const float& width)
{
	auto& manager = GetInstance();
	auto& plantManager = PlantManager::GetInstance();
	switch (manager.m_pointerRenderType)
	{
	case PointerRenderType::Illumination:
		EntityManager::ForEach<GlobalTransform, BranchPointer, Illumination>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [length, width]
		(int i, Entity entity, GlobalTransform& ltw, BranchPointer& c, Illumination& internodeIllumination)
			{
				const glm::vec3 start = ltw.GetPosition();
				const glm::vec3 direction = glm::normalize(internodeIllumination.m_accumulatedDirection);
				const glm::vec3 end = start + direction * length * glm::length(internodeIllumination.m_accumulatedDirection);
				const glm::quat rotation = glm::quatLookAt(direction, glm::vec3(0, 0, 1));
				const glm::mat4 rotationTransform = glm::mat4_cast(rotation * glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f)));
				c.m_value = glm::translate((start + end) / 2.0f) * rotationTransform * glm::scale(glm::vec3(width, glm::distance(start, end) / 2.0f, width));
			}, false
			);
		break;
	case PointerRenderType::Bending:
		EntityManager::ForEach<GlobalTransform, BranchPointer, InternodeGrowth>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [length, width]
		(int i, Entity entity, GlobalTransform& ltw, BranchPointer& c, InternodeGrowth& internodeGrowth)
			{
				const glm::vec3 start = ltw.GetPosition();
				const auto target = internodeGrowth.m_childMeanPosition - internodeGrowth.m_desiredGlobalPosition;
				const glm::vec3 direction = glm::normalize(target);
				const glm::vec3 end = start + direction * length * internodeGrowth.m_MassOfChildren * glm::length(target);
				const glm::quat rotation = glm::quatLookAt(direction, glm::vec3(0, 0, 1));
				const glm::mat4 rotationTransform = glm::mat4_cast(rotation * glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f)));
				c.m_value = glm::translate((start + end) / 2.0f) * rotationTransform * glm::scale(glm::vec3(width, glm::distance(start, end) / 2.0f, width));
			}, false
			);
		break;
	}
}

void TreeManager::UpdateBranchColors()
{
	auto& plantManager = PlantManager::GetInstance();
	auto globalTime = plantManager.m_globalTime;

	auto& manager = GetInstance();
	auto focusingInternode = Entity();
	auto selectedEntity = Entity();
	if (manager.m_currentFocusingInternode.IsValid())
	{
		focusingInternode = manager.m_currentFocusingInternode;
	}
	if (EditorManager::GetInstance().GetSelectedEntity().IsValid())
	{
		selectedEntity = EditorManager::GetInstance().GetSelectedEntity();
	}
#pragma region Process internode color
	switch (GetInstance().m_branchRenderType)
	{
	case BranchRenderType::Illumination:
	{
		EntityManager::ForEach<BranchColor, Illumination, InternodeInfo>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, Illumination& illumination, InternodeInfo& internodeInfo)
			{
				auto& manager = GetInstance();
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto value = illumination.m_currentIntensity;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}
		, false);
	}
	break;
	case BranchRenderType::Inhibitor:
	{
		EntityManager::ForEach<BranchColor, InternodeGrowth>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeGrowth& internodeGrowth)
			{
				auto& manager = GetInstance();
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto value = internodeGrowth.m_inhibitor;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);
	}
	break;
	case BranchRenderType::Sagging:
	{
		EntityManager::ForEach<BranchColor, InternodeGrowth>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeGrowth& internodeGrowth)
			{
				auto& manager = GetInstance();
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto value = internodeGrowth.m_sagging;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);
	}
	break;
	case BranchRenderType::InhibitorTransmitFactor:
	{
		EntityManager::ForEach<BranchColor, InternodeGrowth>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeGrowth& internodeGrowth)
			{
				auto& manager = GetInstance();
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto value = internodeGrowth.m_inhibitorTransmitFactor;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);

	}
	break;
	case BranchRenderType::ResourceToGrow:
	{
		EntityManager::ForEach<BranchColor, InternodeGrowth>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeGrowth& internodeGrowth)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				auto& internodeData = entity.GetPrivateComponent<InternodeData>();
				float totalResource = 0;
				for (const auto& bud : internodeData->m_buds) totalResource += bud.m_currentResource.m_nutrient;
				float value = totalResource;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);

	}
	break;
	case BranchRenderType::Order:
	{
		EntityManager::ForEach<BranchColor, InternodeInfo>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeInfo& internodeInfo)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				float value = internodeInfo.m_order;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);

	}
	break;
	case BranchRenderType::MaxChildOrder:
	{
		EntityManager::ForEach<BranchColor, InternodeStatistics>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeStatistics& internodeStatistics)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				float value = internodeStatistics.m_maxChildOrder;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);

	}
	break;
	case BranchRenderType::Level:
	{
		EntityManager::ForEach<BranchColor, InternodeInfo>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeInfo& internodeInfo)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				float value = internodeInfo.m_level;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);

	}
	break;
	case BranchRenderType::MaxChildLevel:
	{
		EntityManager::ForEach<BranchColor, InternodeStatistics>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeStatistics& internodeStatistics)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				float value = internodeStatistics.m_maxChildLevel;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);

	}
	break;
	case BranchRenderType::IsMaxChild:
	{
		EntityManager::ForEach<BranchColor, InternodeStatistics>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeStatistics& internodeStatistics)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				auto value = internodeStatistics.m_isMaxChild ? 1.0f : 0.2f;
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);
	}
	break;
	case BranchRenderType::ChildrenEndNodeAmount:
	{
		EntityManager::ForEach<BranchColor, InternodeStatistics>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeStatistics& internodeStatistics)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				float value = internodeStatistics.m_childrenEndNodeAmount;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);

	}
	break;
	case BranchRenderType::IsEndNode:
	{
		EntityManager::ForEach<BranchColor, InternodeStatistics>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeStatistics& internodeStatistics)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				auto value = internodeStatistics.m_isEndNode ? 1.0f : 0.2f;
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);
	}
	break;
	case BranchRenderType::DistanceToBranchEnd:
	{
		EntityManager::ForEach<BranchColor, InternodeStatistics>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeStatistics& internodeStatistics)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				float value = internodeStatistics.m_distanceToBranchEnd;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);
	}
	break;
	case BranchRenderType::DistanceToBranchStart:
	{
		EntityManager::ForEach<BranchColor, InternodeStatistics>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeStatistics& internodeStatistics)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				float value = internodeStatistics.m_distanceToBranchStart;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);
	}
	break;
	case BranchRenderType::TotalLength:
	{
		EntityManager::ForEach<BranchColor, InternodeStatistics>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeStatistics& internodeStatistics)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				float value = internodeStatistics.m_totalLength;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);
	}
	break;
	case BranchRenderType::LongestDistanceToAnyEndNode:
	{
		EntityManager::ForEach<BranchColor, InternodeStatistics>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [=]
		(int i, Entity entity, BranchColor& internodeRenderColor, InternodeStatistics& internodeStatistics)
			{
				if (focusingInternode == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 1, 1, 1);
					return;
				}
				if (selectedEntity == entity) {
					internodeRenderColor.m_value = glm::vec4(1, 0, 0, 1);
					return;
				}
				auto& manager = GetInstance();
				float value = internodeStatistics.m_longestDistanceToAnyEndNode;
				if (manager.m_enableBranchDataCompress) value = glm::pow(value, manager.m_branchCompressFactor);
				ColorSet(internodeRenderColor.m_value, value);
			}, false
			);
	}
	break;
	}
#pragma endregion
}

void TreeManager::ColorSet(glm::vec4& target, const float& value)
{
	auto& manager = GetInstance();
	if (manager.m_useColorMap) {
		int compareResult = -1;
		for (int i = 0; i < manager.m_colorMapValues.size(); i++)
		{
			if (value > manager.m_colorMapValues[i]) compareResult = i;
		}
		glm::vec3 color;
		if (compareResult == -1)
		{
			color = manager.m_colorMapColors[0];
		}
		else if (compareResult == manager.m_colorMapValues.size() - 1)
		{
			color = manager.m_colorMapColors.back();
		}
		else
		{
			const auto value1 = manager.m_colorMapValues[compareResult];
			const auto value2 = manager.m_colorMapValues[compareResult + 1];
			const auto color1 = manager.m_colorMapColors[compareResult];
			const auto color2 = manager.m_colorMapColors[compareResult + 1];
			const auto left = value - value1;
			const auto right = value2 - value1;
			color = color1 * left / right + color2 * (1.0f - left / right);
		}
		if (manager.m_useTransparency) target = glm::vec4(color.x, color.y, color.z, manager.m_transparency);
		else target = glm::vec4(color.x, color.y, color.z, 1.0f);
	}
	else
	{
		if (manager.m_useTransparency) target = glm::vec4(value, value, value, manager.m_transparency);
		else target = glm::vec4(value, value, value, 1.0f);
	}
}

void TreeManager::Update()
{
	auto& manager = GetInstance();
	if (manager.m_rightMouseButtonHold && !InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT, WindowManager::GetWindow()))
	{
		manager.m_rightMouseButtonHold = false;
		manager.m_startMouse = false;
	}
	manager.m_internodeDebuggingCamera->ResizeResolution(manager.m_internodeDebuggingCameraResolutionX, manager.m_internodeDebuggingCameraResolutionY);
	manager.m_internodeDebuggingCamera->Clear();

#pragma region Internode debug camera
	CameraComponent::m_cameraInfoBlock.UpdateMatrices(EditorManager::GetInstance().m_sceneCamera.get(),
		EditorManager::GetInstance().m_sceneCameraPosition,
		EditorManager::GetInstance().m_sceneCameraRotation
	);
	CameraComponent::m_cameraInfoBlock.UploadMatrices(EditorManager::GetInstance().m_sceneCamera.get());
	RenderManager::RenderBackGround(manager.m_internodeDebuggingCamera);
#pragma endregion

	auto& plantManager = PlantManager::GetInstance();
	bool needUpdate = false;
	if (plantManager.m_globalTime != manager.m_previousGlobalTime) {
		manager.m_previousGlobalTime = plantManager.m_globalTime;
		if (manager.m_displayTime != manager.m_previousGlobalTime)
		{
			manager.m_displayTime = manager.m_previousGlobalTime;
			needUpdate = true;
		}
	}
#pragma region Rendering
	if (manager.m_drawBranches) {
		if (manager.m_alwaysUpdate || manager.m_updateBranch || needUpdate) {
			manager.m_updateBranch = false;
			UpdateBranchColors();
			UpdateBranchCylinder(manager.m_displayThickness, manager.m_connectionWidth);
		}
		if (manager.m_internodeDebuggingCamera->IsEnabled()) RenderBranchCylinders(manager.m_displayTime);
	}
	if (manager.m_drawPointers) {
		if (manager.m_alwaysUpdate || manager.m_updatePointer || needUpdate) {
			manager.m_updatePointer = false;
			UpdateBranchPointer(manager.m_pointerLength, manager.m_pointerWidth);
		}
		if (manager.m_internodeDebuggingCamera->IsEnabled()) RenderBranchPointers(manager.m_displayTime);
	}
#pragma endregion
}

Entity TreeManager::CreateTree(const Transform& transform)
{
	auto& manager = GetInstance();
	const auto plant = PlantManager::CreatePlant(PlantType::GeneralTree, transform);
	GetLeaves(plant);
	GetRbv(plant);
	EntityManager::SetPrivateComponent(plant, std::make_unique<TreeData>());
	auto material = std::make_shared<Material>();
	material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
	material->m_albedoColor = glm::vec3(0.7f, 0.3f, 0.0f);
	material->m_roughness = 0.01f;
	material->m_metallic = 0.0f;
	material->SetTexture(manager.m_defaultBranchNormalTexture);
	material->SetTexture(manager.m_defaultBranchAlbedoTexture);
	auto meshRenderer = std::make_unique<MeshRenderer>();
	meshRenderer->m_mesh = std::make_shared<Mesh>();
	meshRenderer->m_material = std::move(material);
	

	auto rtt = std::make_unique<RayTracedRenderer>();
	rtt->m_albedoTexture = manager.m_defaultRayTracingBranchAlbedoTexture;
	rtt->m_normalTexture = manager.m_defaultRayTracingBranchNormalTexture;
	rtt->m_mesh = meshRenderer->m_mesh;
	
	plant.SetPrivateComponent(std::move(meshRenderer));
	plant.SetPrivateComponent(std::move(rtt));
	auto& treeData = plant.GetPrivateComponent<TreeData>();
	treeData->m_parameters = TreeParameters();
	return plant;
}

const char* BranchRenderTypes[]{
	"Illumination",
	"Sagging",
	"Inhibitor",
	"InhibitorTransmitFactor",
	"ResourceToGrow",
	"Order",
	"MaxChildOrder",
	"Level",
	"MaxChildLevel",
	"IsMaxChild",
	"ChildrenEndNodeAmount",
	"IsEndNode",
	"DistanceToBranchEnd",
	"DistanceToBranchStart",
	"TotalLength",
	"LongestDistanceToAnyEndNode"

};

const char* PointerRenderTypes[]{
	"Illumination",
	"Bending"
};

void TreeManager::OnGui()
{
	auto& manager = GetInstance();
	auto& plantManager = PlantManager::GetInstance();
#pragma region Main menu
	if (ImGui::BeginMainMenuBar()) {
		if (ImGui::BeginMenu("Tree Manager"))
		{
			ImGui::DragInt("Leaf amount", &manager.m_leafAmount, 0, 0, 50);
			ImGui::DragFloat("Generation radius", &manager.m_radius, 0.01, 0.01, 10);
			ImGui::DragFloat2("Leaf size", &manager.m_leafSize.x, 0.01, 0.01, 10);
			if (ImGui::Button("Regenerate leaves")) GenerateLeavesForTree(PlantManager::GetInstance());
			ImGui::DragFloat("Crown shyness D", &manager.m_crownShynessDiameter, 0.01f, 0.0f, 2.0f);
			if (manager.m_crownShynessDiameter > manager.m_voxelSpaceModule.GetDiameter()) Debug::Error("Diameter too large!");

			if (ImGui::Button("Update metadata"))
			{
				UpdateTreesMetaData(PlantManager::GetInstance());
			}
			if (ImGui::Button("Create...")) {
				ImGui::OpenPopup("New tree wizard");
			}
			ImGui::DragFloat("Mesh resolution", &manager.m_meshResolution, 0.001f, 0, 1);
			ImGui::DragFloat("Mesh subdivision", &manager.m_meshSubdivision, 0.001f, 0, 1);
			if (ImGui::Button("Generate mesh"))
			{
				GenerateMeshForTree(PlantManager::GetInstance());
			}
			FileIO::SaveFile("Save scene as XML", ".xml", [](const std::string& path)
				{
					SerializeScene(path);
				});
			const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
			ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
			if (ImGui::BeginPopupModal("New tree wizard", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
			{
				static std::vector<TreeParameters> newTreeParameters;
				static std::vector<glm::vec3> newTreePositions;
				static std::vector<glm::vec3> newTreeRotations;
				static int newTreeAmount = 1;
				static int currentFocusedNewTreeIndex = 0;
				ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
				ImGui::BeginChild("ChildL", ImVec2(300, 400), true, ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
				if (ImGui::BeginMenuBar())
				{
					if (ImGui::BeginMenu("Settings"))
					{
						static float distance = 10;
						static float variance = 4;
						static float yAxisVar = 180.0f;
						static float xzAxisVar = 0.0f;
						static int expand = 1;
						if (ImGui::BeginMenu("Create forest...")) {
							ImGui::DragFloat("Avg. Y axis rotation", &yAxisVar, 0.01f, 0.0f, 180.0f);
							ImGui::DragFloat("Avg. XZ axis rotation", &xzAxisVar, 0.01f, 0.0f, 90.0f);
							ImGui::DragFloat("Avg. Distance", &distance, 0.01f);
							ImGui::DragFloat("Position variance", &variance, 0.01f);
							ImGui::DragInt("Expand", &expand, 1, 0, 3);
							if (ImGui::Button("Apply"))
							{
								newTreeAmount = (2 * expand + 1) * (2 * expand + 1);
								newTreePositions.resize(newTreeAmount);
								newTreeRotations.resize(newTreeAmount);
								const auto currentSize = newTreeParameters.size();
								newTreeParameters.resize(newTreeAmount);
								for (auto i = currentSize; i < newTreeAmount; i++)
								{
									newTreeParameters[i] = newTreeParameters[0];
								}
								int index = 0;
								for (int i = -expand; i <= expand; i++)
								{
									for (int j = -expand; j <= expand; j++)
									{
										glm::vec3 value = glm::vec3(i * distance, 0, j * distance);
										value.x += glm::linearRand(-variance, variance);
										value.z += glm::linearRand(-variance, variance);
										newTreePositions[index] = value;
										value = glm::vec3(glm::linearRand(-xzAxisVar, xzAxisVar), glm::linearRand(-yAxisVar, yAxisVar), glm::linearRand(-xzAxisVar, xzAxisVar));
										newTreeRotations[index] = value;
										index++;
									}
								}
							}
							ImGui::EndMenu();
						}
						ImGui::InputInt("New Tree Amount", &newTreeAmount);
						if (newTreeAmount < 1) newTreeAmount = 1;
						FileIO::OpenFile("Import parameters for all", ".treeparam", [](const std::string& path)
							{
								newTreeParameters[0].Deserialize(path);
								for (int i = 1; i < newTreeParameters.size(); i++) newTreeParameters[i] = newTreeParameters[0];
							}
						);
						ImGui::EndMenu();
					}
					ImGui::EndMenuBar();
				}
				ImGui::Columns(1);
				if (newTreePositions.size() < newTreeAmount) {
					const auto currentSize = newTreePositions.size();
					newTreeParameters.resize(newTreeAmount);
					for (auto i = currentSize; i < newTreeAmount; i++)
					{
						newTreeParameters[i] = newTreeParameters[0];
					}
					newTreePositions.resize(newTreeAmount);
					newTreeRotations.resize(newTreeAmount);
				}
				for (auto i = 0; i < newTreeAmount; i++)
				{
					std::string title = "New Tree No.";
					title += std::to_string(i);
					const bool opened = ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_NoAutoOpenOnLog | (currentFocusedNewTreeIndex == i ? ImGuiTreeNodeFlags_Framed : ImGuiTreeNodeFlags_FramePadding));
					if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
						currentFocusedNewTreeIndex = i;
					}
					if (opened) {
						ImGui::TreePush();
						ImGui::InputFloat3(("Position##" + std::to_string(i)).c_str(), &newTreePositions[i].x);
						ImGui::TreePop();
					}
				}

				ImGui::EndChild();
				ImGui::PopStyleVar();
				ImGui::SameLine();
				ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
				ImGui::BeginChild("ChildR", ImVec2(400, 400), true, ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar);
				if (ImGui::BeginMenuBar())
				{
					if (ImGui::BeginMenu("Parameters")) {
						FileIO::OpenFile("Import parameters", ".treeparam", [](const std::string& path)
							{
								newTreeParameters[currentFocusedNewTreeIndex].Deserialize(path);
							}
						);
						
						FileIO::SaveFile("Export parameters", ".treeparam", [](const std::string& path)
							{
								newTreeParameters[currentFocusedNewTreeIndex].Serialize(path);
							}
						);
						ImGui::EndMenu();
					}
					ImGui::EndMenuBar();
				}
				ImGui::Columns(1);
				ImGui::PushItemWidth(200);
				newTreeParameters[currentFocusedNewTreeIndex].OnGui();
				ImGui::PopItemWidth();
				ImGui::EndChild();
				ImGui::PopStyleVar();
				ImGui::Separator();
				if (ImGui::Button("OK", ImVec2(120, 0))) {
					//Create tree here.
					for (auto i = 0; i < newTreeAmount; i++) {
						Transform treeTransform;
						treeTransform.SetPosition(newTreePositions[i]);
						treeTransform.SetEulerRotation(glm::radians(newTreeRotations[i]));
						Entity tree = CreateTree(treeTransform);
						tree.SetStatic(true);
						tree.GetPrivateComponent<TreeData>()->m_parameters = newTreeParameters[i];
					}
					ImGui::CloseCurrentPopup();
				}
				ImGui::SetItemDefaultFocus();
				ImGui::SameLine();
				if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
				ImGui::EndPopup();
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}
#pragma endregion
#pragma region Internode debugging camera
	ImVec2 viewPortSize;
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
	ImGui::Begin("Tree Internodes");
	{
		if (ImGui::BeginChild("CameraRenderer", ImVec2(0, 0), false, ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
			if (ImGui::BeginMenuBar())
			{
				if (ImGui::BeginMenu("Settings"))
				{
#pragma region Menu
					ImGui::Checkbox("Force update", &manager.m_alwaysUpdate);
					ImGui::SliderFloat("Display Time", &manager.m_displayTime, 0.0f, plantManager.m_globalTime);
					if (ImGui::ButtonEx("To present", ImVec2(0, 0), manager.m_displayTime != plantManager.m_globalTime ? 0 : ImGuiButtonFlags_Disabled)) manager.m_displayTime = plantManager.m_globalTime;
					if (manager.m_displayTime != plantManager.m_globalTime) {
						ImGui::SameLine();
						if (ImGui::Button("Start from here."))
						{
							ResetTimeForTree(manager.m_displayTime);
						}
					}

					ImGui::Checkbox("Connections", &manager.m_drawBranches);
					if (manager.m_drawBranches)
					{
						if (ImGui::TreeNodeEx("Connection settings", ImGuiTreeNodeFlags_DefaultOpen)) {
							ImGui::Combo("Render type", (int*)&manager.m_branchRenderType, BranchRenderTypes, IM_ARRAYSIZE(BranchRenderTypes));
							ImGui::Checkbox("As transparency", &manager.m_useTransparency);
							if (manager.m_useTransparency) ImGui::SliderFloat("Alpha", &manager.m_transparency, 0, 1);
							ImGui::Checkbox("Compress", &manager.m_enableBranchDataCompress);
							if (manager.m_enableBranchDataCompress) ImGui::DragFloat("Compress factor", &manager.m_branchCompressFactor, 0.01f, 0.01f, 1.0f);
							ImGui::Checkbox("Color Map", &manager.m_useColorMap);
							if (manager.m_useColorMap) {
								static int savedAmount = 3;
								ImGui::SliderInt("Slot amount", &manager.m_colorMapSegmentAmount, 2, 10);
								if (savedAmount != manager.m_colorMapSegmentAmount)
								{
									manager.m_colorMapValues.resize(manager.m_colorMapSegmentAmount);
									manager.m_colorMapColors.resize(manager.m_colorMapSegmentAmount);
									for (int i = 0; i < manager.m_colorMapSegmentAmount; i++)
									{
										if (i != 0 && manager.m_colorMapValues[i] < manager.m_colorMapValues[i - 1]) manager.m_colorMapValues[i] = manager.m_colorMapValues[i - 1] + 1;
										if (i >= savedAmount)manager.m_colorMapColors[i] = glm::linearRand(glm::vec3(0.0f), glm::vec3(1.0f));
									}
									savedAmount = manager.m_colorMapSegmentAmount;
								}
								for (int i = 0; i < manager.m_colorMapValues.size(); i++)
								{
									if (i == 0)
									{
										ImGui::DragFloat("Value 0", &manager.m_colorMapValues[0], 0.1f, 0.0f, manager.m_colorMapValues[1]);
										ImGui::ColorEdit3("Color 0", &manager.m_colorMapColors[0].x);
									}
									else if (i == manager.m_colorMapValues.size() - 1)
									{
										ImGui::DragFloat(("Value" + std::to_string(i)).c_str(), &manager.m_colorMapValues[i], 0.1f, manager.m_colorMapValues[i - 1] + 0.1f, 9999.0f);
										ImGui::ColorEdit3(("Color" + std::to_string(i)).c_str(), &manager.m_colorMapColors[i].x);
									}
									else
									{
										ImGui::DragFloat(("Value" + std::to_string(i)).c_str(), &manager.m_colorMapValues[i], 0.1f, manager.m_colorMapValues[i - 1] + 0.1f, manager.m_colorMapValues[i + 1]);
										ImGui::ColorEdit3(("Color" + std::to_string(i)).c_str(), &manager.m_colorMapColors[i].x);
									}
								}
							}

							if (ImGui::Checkbox("Display thickness", &manager.m_displayThickness)) manager.m_updateBranch = true;
							if (!manager.m_displayThickness) if (ImGui::DragFloat("Connection width", &manager.m_connectionWidth, 0.01f, 0.01f, 1.0f)) manager.m_updateBranch = true;
							ImGui::TreePop();
						}
					}
					ImGui::Checkbox("Pointers", &manager.m_drawPointers);
					if (manager.m_drawPointers)
					{
						if (ImGui::TreeNodeEx("Pointer settings", ImGuiTreeNodeFlags_DefaultOpen)) {
							ImGui::Combo("Render type", (int*)&manager.m_pointerRenderType, PointerRenderTypes, IM_ARRAYSIZE(PointerRenderTypes));
							ImGui::Checkbox("Compress", &manager.m_enablePointerDataCompress);
							if (manager.m_pointerCompressFactor) ImGui::DragFloat("Compress factor", &manager.m_branchCompressFactor, 0.01f, 0.01f, 1.0f);
							if (ImGui::ColorEdit4("Pointer color", &manager.m_pointerColor.x)) manager.m_updatePointer = true;
							if (ImGui::DragFloat("Pointer length", &manager.m_pointerLength, 0.01f, 0.01f, 3.0f)) manager.m_updatePointer = true;
							if (ImGui::DragFloat("Pointer width", &manager.m_pointerWidth, 0.01f, 0.01f, 1.0f)) manager.m_updatePointer = true;
							ImGui::TreePop();
						}
					}
					manager.m_voxelSpaceModule.OnGui();

#pragma endregion
					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}
			viewPortSize = ImGui::GetWindowSize();
			viewPortSize.y -= 20;
			if (viewPortSize.y < 0) viewPortSize.y = 0;
			manager.m_internodeDebuggingCameraResolutionX = viewPortSize.x;
			manager.m_internodeDebuggingCameraResolutionY = viewPortSize.y;
			ImGui::Image(reinterpret_cast<ImTextureID>(manager.m_internodeDebuggingCamera->GetTexture()->Texture()->Id()), viewPortSize, ImVec2(0, 1), ImVec2(1, 0));
			glm::vec2 mousePosition = glm::vec2(FLT_MAX, FLT_MIN);
			if (ImGui::IsWindowFocused())
			{
				bool valid = true;
				mousePosition = InputManager::GetMouseAbsolutePositionInternal(WindowManager::GetWindow());
				float xOffset = 0;
				float yOffset = 0;
				if (valid) {
					if (!manager.m_startMouse) {
						manager.m_lastX = mousePosition.x;
						manager.m_lastY = mousePosition.y;
						manager.m_startMouse = true;
					}
					xOffset = mousePosition.x - manager.m_lastX;
					yOffset = -mousePosition.y + manager.m_lastY;
					manager.m_lastX = mousePosition.x;
					manager.m_lastY = mousePosition.y;
#pragma region Scene Camera Controller
					if (!manager.m_rightMouseButtonHold && InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT, WindowManager::GetWindow())) {
						manager.m_rightMouseButtonHold = true;
					}
					if (manager.m_rightMouseButtonHold && !EditorManager::GetInstance().m_lockCamera)
					{
						glm::vec3 front = EditorManager::GetInstance().m_sceneCameraRotation * glm::vec3(0, 0, -1);
						glm::vec3 right = EditorManager::GetInstance().m_sceneCameraRotation * glm::vec3(1, 0, 0);
						if (InputManager::GetKeyInternal(GLFW_KEY_W, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition += front * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime()) * EditorManager::GetInstance().m_velocity;
						}
						if (InputManager::GetKeyInternal(GLFW_KEY_S, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition -= front * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime()) * EditorManager::GetInstance().m_velocity;
						}
						if (InputManager::GetKeyInternal(GLFW_KEY_A, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition -= right * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime()) * EditorManager::GetInstance().m_velocity;
						}
						if (InputManager::GetKeyInternal(GLFW_KEY_D, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition += right * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime()) * EditorManager::GetInstance().m_velocity;
						}
						if (InputManager::GetKeyInternal(GLFW_KEY_LEFT_SHIFT, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition.y += EditorManager::GetInstance().m_velocity * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime());
						}
						if (InputManager::GetKeyInternal(GLFW_KEY_LEFT_CONTROL, WindowManager::GetWindow())) {
							EditorManager::GetInstance().m_sceneCameraPosition.y -= EditorManager::GetInstance().m_velocity * static_cast<float>(Application::GetCurrentWorld()->Time()->DeltaTime());
						}
						if (xOffset != 0.0f || yOffset != 0.0f) {
							EditorManager::GetInstance().m_sceneCameraYawAngle += xOffset * EditorManager::GetInstance().m_sensitivity;
							EditorManager::GetInstance().m_sceneCameraPitchAngle += yOffset * EditorManager::GetInstance().m_sensitivity;
							if (EditorManager::GetInstance().m_sceneCameraPitchAngle > 89.0f)
								EditorManager::GetInstance().m_sceneCameraPitchAngle = 89.0f;
							if (EditorManager::GetInstance().m_sceneCameraPitchAngle < -89.0f)
								EditorManager::GetInstance().m_sceneCameraPitchAngle = -89.0f;

							EditorManager::GetInstance().m_sceneCameraRotation = CameraComponent::ProcessMouseMovement(EditorManager::GetInstance().m_sceneCameraYawAngle, EditorManager::GetInstance().m_sceneCameraPitchAngle, false);
						}
					}
#pragma endregion
					if (manager.m_drawBranches) {
#pragma region Ray selection
						manager.m_currentFocusingInternode = Entity();
						std::mutex writeMutex;
						auto windowPos = ImGui::GetWindowPos();
						auto windowSize = ImGui::GetWindowSize();
						mousePosition.x -= windowPos.x;
						mousePosition.x -= windowSize.x;
						mousePosition.y -= windowPos.y + 20;
						float minDistance = FLT_MAX;
						GlobalTransform cameraLtw;
						cameraLtw.m_value = glm::translate(EditorManager::GetInstance().m_sceneCameraPosition) * glm::mat4_cast(EditorManager::GetInstance().m_sceneCameraRotation);
						const Ray cameraRay = manager.m_internodeDebuggingCamera->ScreenPointToRay(
							cameraLtw, mousePosition);
						EntityManager::ForEach<GlobalTransform, BranchCylinderWidth>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery, [&, cameraLtw, cameraRay](int i, Entity entity, GlobalTransform& ltw, BranchCylinderWidth& width)
							{
								const glm::vec3 position = ltw.m_value[3];
								const auto parentPosition = EntityManager::GetParent(entity).GetComponentData<GlobalTransform>().GetPosition();
								const auto center = (position + parentPosition) / 2.0f;
								auto& dir = cameraRay.m_direction;
								auto& pos = cameraRay.m_start;
								const auto radius = width.m_value;
								const auto height = glm::distance(parentPosition, position);

								if (!cameraRay.Intersect(center, height / 2.0f)) return;
#pragma region Line Line intersection
								/*
								 * http://geomalgorithms.com/a07-_distance.html
								 */
								glm::vec3 u = pos - (pos + dir);
								glm::vec3 v = position - parentPosition;
								glm::vec3 w = (pos + dir) - parentPosition;
								const auto a = dot(u, u);        // always >= 0
								const auto b = dot(u, v);
								const auto c = dot(v, v);        // always >= 0
								const auto d = dot(u, w);
								const auto e = dot(v, w);
								const auto dotP = a * c - b * b;       // always >= 0
								float sc, tc;
								// compute the line parameters of the two closest points
								if (dotP < 0.001f) {          // the lines are almost parallel
									sc = 0.0f;
									tc = (b > c ? d / b : e / c);   // use the largest denominator
								}
								else {
									sc = (b * e - c * d) / dotP;
									tc = (a * e - b * d) / dotP;
								}
								// get the difference of the two closest points
								glm::vec3 dP = w + sc * u - tc * v;  // =  L1(sc) - L2(tc)
								if (glm::length(dP) > radius) return;
#pragma endregion
								const auto distance = glm::distance(glm::vec3(cameraLtw.m_value[3]), glm::vec3(center));
								std::lock_guard<std::mutex> lock(writeMutex);
								if (distance < minDistance)
								{
									minDistance = distance;
									manager.m_currentFocusingInternode = entity;
								}

							}
						);
						if (InputManager::GetMouseInternal(GLFW_MOUSE_BUTTON_LEFT, WindowManager::GetWindow()))
						{
							if (!manager.m_currentFocusingInternode.IsNull())
							{
								EditorManager::SetSelectedEntity(manager.m_currentFocusingInternode);
							}
						}
#pragma endregion
					}
				}
			}
		}
		ImGui::EndChild();
		auto* window = ImGui::FindWindowByName("Tree Internodes");
		manager.m_internodeDebuggingCamera->SetEnabled(!(window->Hidden && !window->Collapsed));
	}
	ImGui::End();
	ImGui::PopStyleVar();

#pragma endregion
}

void TreeManager::RenderBranchCylinders(const float& displayTime)
{
	std::vector<BranchCylinder> branchCylinders;
	auto& plantManager = PlantManager::GetInstance();
	auto& manager = GetInstance();
	plantManager.m_internodeQuery.ToComponentDataArray<BranchCylinder, InternodeInfo>(branchCylinders, [displayTime](const InternodeInfo& internodeInfo)
		{
			return internodeInfo.m_startGlobalTime <= displayTime;
		});
	std::vector<BranchColor> branchColors;
	plantManager.m_internodeQuery.ToComponentDataArray<BranchColor, InternodeInfo>(branchColors, [displayTime](const InternodeInfo& internodeInfo)
		{
			return internodeInfo.m_startGlobalTime <= displayTime;
		});
	if (!branchCylinders.empty())RenderManager::DrawGizmoMeshInstancedColored(
		DefaultResources::Primitives::Cylinder.get(), manager.m_internodeDebuggingCamera.get(), EditorManager::GetInstance().m_sceneCameraPosition, EditorManager::GetInstance().m_sceneCameraRotation,
		(glm::vec4*)branchColors.data(), (glm::mat4*)branchCylinders.data(), branchCylinders.size(), glm::mat4(1.0f), 1.0f);

}

void TreeManager::RenderBranchPointers(const float& displayTime)
{
	std::vector<BranchPointer> branchPointers;
	auto& plantManager = PlantManager::GetInstance();
	auto& manager = GetInstance();
	plantManager.m_internodeQuery.ToComponentDataArray<BranchPointer, InternodeInfo>(branchPointers, [displayTime](const InternodeInfo& internodeInfo)
		{
			return internodeInfo.m_startGlobalTime <= displayTime;
		});
	if (!branchPointers.empty())RenderManager::DrawGizmoMeshInstanced(
		DefaultResources::Primitives::Cylinder.get(), manager.m_internodeDebuggingCamera.get(), EditorManager::GetInstance().m_sceneCameraPosition, EditorManager::GetInstance().m_sceneCameraRotation
		, manager.m_pointerColor, reinterpret_cast<glm::mat4*>(branchPointers.data()), branchPointers.size(), glm::mat4(1.0f), 1.0f);

}

TreeManager& TreeManager::GetInstance()
{
	static TreeManager instance;
	return instance;
}

void TreeManager::SimpleMeshGenerator(Entity& internode, std::vector<Vertex>& vertices,
	std::vector<unsigned>& indices, const glm::vec3& normal, float resolution, int parentStep)
{
	glm::vec3 newNormalDir = normal;
	const glm::vec3 front = internode.GetComponentData<InternodeGrowth>().m_desiredGlobalRotation * glm::vec3(0.0f, 0.0f, -1.0f);
	newNormalDir = glm::cross(glm::cross(front, newNormalDir), front);

	auto& list = EntityManager::GetPrivateComponent<InternodeData>(internode);
	if (list->m_rings.empty())
	{
		return;
	}
	auto step = list->m_step;
	//For stitching
	const int pStep = parentStep > 0 ? parentStep : step;

	list->m_normalDir = newNormalDir;
	float angleStep = 360.0f / static_cast<float>(pStep);
	int vertexIndex = vertices.size();
	Vertex archetype;
	float textureXStep = 1.0f / pStep * 4.0f;
	for (int i = 0; i < pStep; i++) {
		archetype.m_position = list->m_rings.at(0).GetPoint(newNormalDir, angleStep * i, true);
		const float x = i < pStep / 2 ? i * textureXStep : (pStep - i) * textureXStep;
		archetype.m_texCoords = glm::vec2(x, 0.0f);
		vertices.push_back(archetype);
	}
	std::vector<float> angles;
	angles.resize(step);
	std::vector<float> pAngles;
	pAngles.resize(pStep);

	for (auto i = 0; i < pStep; i++) {
		pAngles[i] = angleStep * i;
	}
	angleStep = 360.0f / static_cast<float>(step);
	for (auto i = 0; i < step; i++) {
		angles[i] = angleStep * i;
	}

	std::vector<unsigned> pTarget;
	std::vector<unsigned> target;
	pTarget.resize(pStep);
	target.resize(step);
	for (int i = 0; i < pStep; i++) {
		//First we allocate nearest vertices for parent.
		auto minAngleDiff = 360.0f;
		for (auto j = 0; j < step; j++) {
			const float diff = glm::abs(pAngles[i] - angles[j]);
			if (diff < minAngleDiff) {
				minAngleDiff = diff;
				pTarget[i] = j;
			}
		}
	}
	for (int i = 0; i < step; i++) {
		//Second we allocate nearest vertices for child
		float minAngleDiff = 360.0f;
		for (int j = 0; j < pStep; j++) {
			const float diff = glm::abs(angles[i] - pAngles[j]);
			if (diff < minAngleDiff) {
				minAngleDiff = diff;
				target[i] = j;
			}
		}
	}
	for (int i = 0; i < pStep; i++) {
		if (pTarget[i] == pTarget[i == pStep - 1 ? 0 : i + 1]) {
			indices.push_back(vertexIndex + i);
			indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
			indices.push_back(vertexIndex + pStep + pTarget[i]);
		}
		else {
			indices.push_back(vertexIndex + i);
			indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
			indices.push_back(vertexIndex + pStep + pTarget[i]);

			indices.push_back(vertexIndex + pStep + pTarget[i == pStep - 1 ? 0 : i + 1]);
			indices.push_back(vertexIndex + pStep + pTarget[i]);
			indices.push_back(vertexIndex + (i == pStep - 1 ? 0 : i + 1));
		}
	}

	vertexIndex += pStep;
	textureXStep = 1.0f / step * 4.0f;
	const int ringSize = list->m_rings.size();
	for (auto ringIndex = 0; ringIndex < ringSize; ringIndex++) {
		for (auto i = 0; i < step; i++) {
			archetype.m_position = list->m_rings.at(ringIndex).GetPoint(newNormalDir, angleStep * i, false);
			const auto x = i < (step / 2) ? i * textureXStep : (step - i) * textureXStep;
			const auto y = ringIndex % 2 == 0 ? 1.0f : 0.0f;
			archetype.m_texCoords = glm::vec2(x, y);
			vertices.push_back(archetype);
		}
		if (ringIndex != 0) {
			for (int i = 0; i < step - 1; i++) {
				//Down triangle
				indices.push_back(vertexIndex + (ringIndex - 1) * step + i);
				indices.push_back(vertexIndex + (ringIndex - 1) * step + i + 1);
				indices.push_back(vertexIndex + (ringIndex)*step + i);
				//Up triangle
				indices.push_back(vertexIndex + (ringIndex)*step + i + 1);
				indices.push_back(vertexIndex + (ringIndex)*step + i);
				indices.push_back(vertexIndex + (ringIndex - 1) * step + i + 1);
			}
			//Down triangle
			indices.push_back(vertexIndex + (ringIndex - 1) * step + step - 1);
			indices.push_back(vertexIndex + (ringIndex - 1) * step);
			indices.push_back(vertexIndex + (ringIndex)*step + step - 1);
			//Up triangle
			indices.push_back(vertexIndex + (ringIndex)*step);
			indices.push_back(vertexIndex + (ringIndex)*step + step - 1);
			indices.push_back(vertexIndex + (ringIndex - 1) * step);
		}
	}

	EntityManager::ForEachChild(internode, [&vertices, &indices, &newNormalDir, resolution, step](Entity child)
		{
			SimpleMeshGenerator(child, vertices, indices, newNormalDir, resolution, step);
		}
	);
}

void TreeManager::GenerateMeshForTree(PlantManager& manager)
{
	auto& treeManager = GetInstance();
	if (treeManager.m_meshResolution <= 0.0f) {
		Debug::Error("TreeManager: Resolution must be larger than 0!");
		return;
	}

	//Prepare ring mesh.
	EntityManager::ForEach<GlobalTransform, Transform, InternodeGrowth, InternodeInfo>(JobManager::PrimaryWorkers(), manager.m_internodeQuery, [&](int i, Entity internode, GlobalTransform& globalTransform, Transform& transform, InternodeGrowth& internodeGrowth, InternodeInfo& internodeInfo)
		{
			if (internodeInfo.m_plantType != PlantType::GeneralTree) return;
			const Entity parent = EntityManager::GetParent(internode);
			if (parent == internodeInfo.m_plant) return;
			bool isRootInternode = false;
			if (EntityManager::GetParent(parent) == internodeInfo.m_plant) isRootInternode = true;

			auto& list = EntityManager::GetPrivateComponent<InternodeData>(internode);
			list->m_rings.clear();
			glm::mat4 treeTransform = internodeInfo.m_plant.GetComponentData<GlobalTransform>().m_value;
			GlobalTransform parentGlobalTransform;
			parentGlobalTransform.m_value = glm::inverse(treeTransform) * parent.GetComponentData<GlobalTransform>().m_value;
			float parentThickness = isRootInternode ? internodeGrowth.m_thickness * 1.25f : parent.GetComponentData<InternodeGrowth>().m_thickness;
			glm::vec3 parentScale;
			glm::quat parentRotation;
			glm::vec3 parentTranslation;
			parentGlobalTransform.Decompose(parentTranslation, parentRotation, parentScale);

			glm::vec3 scale;
			glm::quat rotation;
			glm::vec3 translation;
			GlobalTransform copyGT;
			copyGT.m_value = glm::inverse(treeTransform) * globalTransform.m_value;
			copyGT.Decompose(translation, rotation, scale);

			glm::vec3 parentDir = isRootInternode ? glm::vec3(0, 1, 0) : parentRotation * glm::vec3(0, 0, -1);
			glm::vec3 dir = rotation * glm::vec3(0, 0, -1);
			glm::quat mainChildRotation = rotation;
			if (!internodeGrowth.m_thickestChild.IsNull())
			{
				GlobalTransform thickestChildTransform;
				thickestChildTransform.m_value = glm::inverse(treeTransform) * internodeGrowth.m_thickestChild.GetComponentData<GlobalTransform>().m_value;
				mainChildRotation = thickestChildTransform.GetRotation();
			}
			glm::vec3 mainChildDir = mainChildRotation * glm::vec3(0, 0, -1);
			GlobalTransform parentThickestChildGlobalTransform;
			parentThickestChildGlobalTransform.m_value = glm::inverse(treeTransform) * parent.GetComponentData<InternodeGrowth>().m_thickestChild.GetComponentData<GlobalTransform>().m_value;
			glm::vec3 parentMainChildDir = parentThickestChildGlobalTransform.GetRotation() * glm::vec3(0, 0, -1);
			glm::vec3 fromDir = isRootInternode ? parentDir : (parentDir + parentMainChildDir) / 2.0f;
			dir = (dir + mainChildDir) / 2.0f;
#pragma region Subdivision internode here.
			auto distance = glm::distance(parentTranslation, translation);

			int step = parentThickness / treeManager.m_meshResolution;
			if (step < 4) step = 4;
			if (step % 2 != 0) step++;
			list->m_step = step;
			int amount = static_cast<int>(0.5f + distance * treeManager.m_meshSubdivision);
			if (amount % 2 != 0) amount++;
			BezierCurve curve = BezierCurve(parentTranslation, parentTranslation + distance / 3.0f * fromDir, translation - distance / 3.0f * dir, translation);
			float posStep = 1.0f / static_cast<float>(amount);
			glm::vec3 dirStep = (dir - fromDir) / static_cast<float>(amount);
			float radiusStep = (internodeGrowth.m_thickness - parentThickness) / static_cast<float>(amount);

			for (int i = 1; i < amount; i++) {
				float startThickness = static_cast<float>(i - 1) * radiusStep;
				float endThickness = static_cast<float>(i) * radiusStep;
				list->m_rings.emplace_back(
					curve.GetPoint(posStep * (i - 1)), curve.GetPoint(posStep * i),
					fromDir + static_cast<float>(i - 1) * dirStep, fromDir + static_cast<float>(i) * dirStep,
					parentThickness + startThickness, parentThickness + endThickness);
			}
			if (amount > 1)list->m_rings.emplace_back(curve.GetPoint(1.0f - posStep), translation, dir - dirStep, dir, internodeGrowth.m_thickness - radiusStep, internodeGrowth.m_thickness);
			else list->m_rings.emplace_back(parentTranslation, translation, fromDir, dir, parentThickness, internodeGrowth.m_thickness);
#pragma endregion
		}

	);
	for (const auto& plant : manager.m_plants) {
		if (plant.GetComponentData<PlantInfo>().m_plantType != PlantType::GeneralTree) continue;
		if (!plant.HasPrivateComponent<MeshRenderer>() || !plant.HasPrivateComponent<TreeData>()) continue;
		auto& meshRenderer = EntityManager::GetPrivateComponent<MeshRenderer>(plant);
		auto& treeData = EntityManager::GetPrivateComponent<TreeData>(plant);
		if (Entity rootInternode = GetRootInternode(plant); !rootInternode.IsNull())
		{
			if (EntityManager::GetChildrenAmount(rootInternode) != 0) {
				std::vector<unsigned> indices;
				std::vector<Vertex> vertices;
				SimpleMeshGenerator(EntityManager::GetChildren(rootInternode).at(0), vertices, indices, glm::vec3(0, 0, 1), treeManager.m_meshResolution);
				meshRenderer->m_mesh->SetVertices(17, vertices, indices);
				treeData->m_meshGenerated = true;
			}
		}
	}
}

void TreeManager::GenerateLeavesForTree(PlantManager& plantManager)
{
	auto& manager = GetInstance();
	for (const auto& plant : PlantManager::GetInstance().m_plants)
	{
		GetLeaves(plant).GetPrivateComponent<TreeLeaves>()->m_transforms.clear();
	}
	std::mutex mutex;
	EntityManager::ForEach<GlobalTransform, InternodeInfo, InternodeStatistics, Illumination>(JobManager::PrimaryWorkers(), plantManager.m_internodeQuery,
		[&](int index, Entity internode, GlobalTransform& globalTransform, InternodeInfo& internodeInfo, InternodeStatistics& internodeStatistics, Illumination& internodeIllumination)
		{
			if (!internodeInfo.m_plant.IsEnabled()) return;
			if (internodeInfo.m_plantType != PlantType::GeneralTree) return;
			if (internodeStatistics.m_longestDistanceToAnyEndNode > 0.5f) return;

			auto& treeLeaves = GetLeaves(internodeInfo.m_plant).GetPrivateComponent<TreeLeaves>();
			auto& internodeData = internode.GetPrivateComponent<InternodeData>();
			internodeData->m_leavesTransforms.clear();
			const glm::quat rotation = globalTransform.GetRotation();
			const glm::vec3 left = rotation * glm::vec3(1, 0, 0);
			const glm::vec3 right = rotation * glm::vec3(-1, 0, 0);
			const glm::vec3 up = rotation * glm::vec3(0, 1, 0);
			std::lock_guard lock(mutex);
			for (int i = 0; i < manager.m_leafAmount; i++)
			{
				const auto transform = globalTransform.m_value *
					(
						glm::translate(glm::linearRand(glm::vec3(-manager.m_radius), glm::vec3(manager.m_radius))) * glm::mat4_cast(glm::quat(glm::radians(glm::linearRand(glm::vec3(0.0f), glm::vec3(360.0f))))) * glm::scale(glm::vec3(manager.m_leafSize.x, 1.0f, manager.m_leafSize.y))
						);
				internodeData->m_leavesTransforms.push_back(transform);
				treeLeaves->m_transforms.push_back(transform);
			}
			/*
			internodeData->m_leavesTransforms.push_back(globalTransform.m_value *
				(
					glm::translate(left * 0.1f) * glm::mat4_cast(glm::quatLookAt(-left, up)) * glm::scale(glm::vec3(0.1f))
					)
			);
			internodeData->m_leavesTransforms.push_back(globalTransform.m_value *
				(
					glm::translate(right * 0.1f) * glm::mat4_cast(glm::quatLookAt(-right, up)) * glm::scale(glm::vec3(0.1f))
					)
			);
			*/
		}
	);
	for (const auto& plant : PlantManager::GetInstance().m_plants)
	{
		GetLeaves(plant).GetPrivateComponent<TreeLeaves>()->FormMesh();
	}
}

void TreeManager::FormCandidates(PlantManager& manager,
	std::vector<InternodeCandidate>& candidates)
{
	const float globalTime = manager.m_globalTime;
	std::mutex mutex;
	EntityManager::ForEach<GlobalTransform, Transform, InternodeInfo, InternodeGrowth, InternodeStatistics, Illumination>(JobManager::PrimaryWorkers(), PlantManager::GetInstance().m_internodeQuery,
		[&, globalTime](int index, Entity internode, GlobalTransform& globalTransform, Transform& transform, InternodeInfo& internodeInfo, InternodeGrowth& internodeGrowth, InternodeStatistics& internodeStatistics, Illumination& internodeIllumination)
		{
			if (internodeInfo.m_plantType != PlantType::GeneralTree) return;
			auto& treeData = internodeInfo.m_plant.GetPrivateComponent<TreeData>();
			auto plantInfo = internodeInfo.m_plant.GetComponentData<PlantInfo>();
			if (!internodeInfo.m_plant.IsEnabled()) return;
			auto& internodeData = internode.GetPrivateComponent<InternodeData>();
#pragma region Go through each bud
			for (int i = 0; i < internodeData->m_buds.size(); i++)
			{
				auto& bud = internodeData->m_buds[i];
				if (!bud.m_active || !bud.m_enoughForGrowth) continue;
				bud.m_active = false;
				bud.m_deathGlobalTime = globalTime;
				const bool& isApical = bud.m_isApical;
#pragma region Form candidate
				glm::quat prevGlobalRotation = globalTransform.GetRotation();
				auto candidate = InternodeCandidate();
				candidate.m_owner = internodeData->m_owner;
				candidate.m_parent = internode;


				candidate.m_info.m_startGlobalTime = globalTime;
				candidate.m_info.m_plant = internodeInfo.m_plant;
				candidate.m_info.m_startAge = plantInfo.m_age;
				candidate.m_info.m_order = internodeInfo.m_order + (isApical ? 0 : 1);
				candidate.m_info.m_level = internodeInfo.m_level + (isApical ? 0 : 1);

				candidate.m_growth.m_distanceToRoot = internodeGrowth.m_distanceToRoot + 1;
				candidate.m_growth.m_inhibitorTransmitFactor = GetGrowthParameter(GrowthParameterType::InhibitorTransmitFactor, treeData, internodeInfo, internodeGrowth, internodeStatistics);
				glm::quat desiredRotation = glm::radians(glm::vec3(0.0f, 0.0f, 0.0f)); //Apply main angle
				glm::vec3 up = glm::vec3(0, 1, 0);
				up = glm::rotate(up, glm::radians(bud.m_mainAngle), glm::vec3(0, 0, -1));
				if (!bud.m_isApical) {
					desiredRotation = glm::rotate(desiredRotation, GetGrowthParameter(GrowthParameterType::BranchingAngle, treeData, internodeInfo, internodeGrowth, internodeStatistics), desiredRotation * up); //Apply branching angle
				}
				desiredRotation = glm::rotate(desiredRotation, GetGrowthParameter(GrowthParameterType::RollAngle, treeData, internodeInfo, internodeGrowth, internodeStatistics), desiredRotation * glm::vec3(0, 0, -1)); //Apply roll angle
				desiredRotation = glm::rotate(desiredRotation, GetGrowthParameter(GrowthParameterType::ApicalAngle, treeData, internodeInfo, internodeGrowth, internodeStatistics), desiredRotation * glm::vec3(0, 1, 0)); //Apply apical angle
#pragma region Apply tropisms
				glm::quat globalDesiredRotation = prevGlobalRotation * desiredRotation;
				glm::vec3 desiredFront = globalDesiredRotation * glm::vec3(0.0f, 0.0f, -1.0f);
				glm::vec3 desiredUp = globalDesiredRotation * glm::vec3(0.0f, 1.0f, 0.0f);
				PlantManager::ApplyTropism(-treeData->m_gravityDirection, GetGrowthParameter(GrowthParameterType::Gravitropism, treeData, internodeInfo, internodeGrowth, internodeStatistics), desiredFront, desiredUp);
				if (internodeIllumination.m_accumulatedDirection != glm::vec3(0.0f)) PlantManager::ApplyTropism(glm::normalize(internodeIllumination.m_accumulatedDirection), GetGrowthParameter(GrowthParameterType::Phototropism, treeData, internodeInfo, internodeGrowth, internodeStatistics), desiredFront, desiredUp);
				globalDesiredRotation = glm::quatLookAt(desiredFront, desiredUp);
				desiredRotation = glm::inverse(prevGlobalRotation) * globalDesiredRotation;
#pragma endregion

#pragma region Calculate transform
				glm::quat globalRotation = globalTransform.GetRotation() * candidate.m_growth.m_desiredLocalRotation;
				glm::vec3 front = globalRotation * glm::vec3(0, 0, -1);
				glm::vec3 positionDelta = front * treeData->m_parameters.m_internodeLengthBase;
				glm::vec3 newInternodePosition = globalTransform.GetPosition() + positionDelta;
				candidate.m_globalTransform.m_value = glm::translate(newInternodePosition)
					* glm::mat4_cast(globalRotation) * glm::scale(glm::vec3(1.0f));
				candidate.m_transform.m_value = glm::inverse(globalTransform.m_value) * candidate.m_globalTransform.m_value;
#pragma endregion

				candidate.m_growth.m_desiredLocalRotation = desiredRotation;

				candidate.m_statistics.m_isEndNode = true;
				candidate.m_buds = std::vector<Bud>();

				Bud apicalBud;
				float totalResourceWeight = 0.0f;
				apicalBud.m_isApical = true;
				apicalBud.m_resourceWeight = GetGrowthParameter(GrowthParameterType::ResourceWeightApical, treeData, internodeInfo, internodeGrowth, internodeStatistics);
				totalResourceWeight += apicalBud.m_resourceWeight;
				apicalBud.m_avoidanceAngle = GetGrowthParameter(GrowthParameterType::AvoidanceAngle, treeData, internodeInfo, internodeGrowth, internodeStatistics);
				candidate.m_buds.push_back(apicalBud);
				const auto budAmount = GetGrowthParameter(GrowthParameterType::LateralBudPerNode, treeData, internodeInfo, internodeGrowth, internodeStatistics);
				for (int budIndex = 0; budIndex < budAmount; budIndex++) {
					Bud lateralBud;
					lateralBud.m_isApical = false;
					lateralBud.m_resourceWeight = GetGrowthParameter(GrowthParameterType::ResourceWeightLateral, treeData, internodeInfo, internodeGrowth, internodeStatistics);
					totalResourceWeight += lateralBud.m_resourceWeight;
					lateralBud.m_mainAngle = 360.0f * (glm::gaussRand(1.0f, 0.5f) + budIndex) / budAmount;
					lateralBud.m_avoidanceAngle = GetGrowthParameter(GrowthParameterType::AvoidanceAngle, treeData, internodeInfo, internodeGrowth, internodeStatistics);
					candidate.m_buds.push_back(lateralBud);
				}
#pragma region Calculate resource weight for new buds and transport resource from current bud to new buds.
				ResourceParcel& currentBudResourceParcel = bud.m_currentResource;
				auto consumer = ResourceParcel(-1.0f, -1.0f);
				consumer.m_globalTime = globalTime;
				currentBudResourceParcel += consumer;
				bud.m_resourceLog.push_back(consumer);
				for (auto& newBud : candidate.m_buds) {
					newBud.m_resourceWeight /= totalResourceWeight;
					auto resourceParcel = ResourceParcel(
						currentBudResourceParcel.m_nutrient * newBud.m_resourceWeight,
						currentBudResourceParcel.m_carbon * newBud.m_resourceWeight);
					resourceParcel.m_globalTime = globalTime;
					newBud.m_currentResource += resourceParcel;
					newBud.m_resourceLog.push_back(resourceParcel);
				}
				auto resourceParcel = ResourceParcel(
					-currentBudResourceParcel.m_nutrient,
					-currentBudResourceParcel.m_carbon);
				resourceParcel.m_globalTime = globalTime;
				bud.m_currentResource += resourceParcel;
				bud.m_resourceLog.push_back(resourceParcel);
#pragma endregion
				std::lock_guard lock(mutex);
				candidates.push_back(std::move(candidate));
#pragma endregion
			}
#pragma endregion
		}, false
		);
}

float TreeManager::GetGrowthParameter(const GrowthParameterType& type,
	std::unique_ptr<TreeData>& treeData, InternodeInfo& internodeInfo, InternodeGrowth& internodeGrowth,
	InternodeStatistics& internodeStatistics)
{
	float value = 0;
	switch (type)
	{
	case GrowthParameterType::InhibitorTransmitFactor:
		value = treeData->m_parameters.m_inhibitorDistanceFactor;
		break;
	case GrowthParameterType::Gravitropism:
		value = treeData->m_parameters.m_gravitropism;
		break;
	case GrowthParameterType::Phototropism:
		value = treeData->m_parameters.m_phototropism;
		break;
	case GrowthParameterType::BranchingAngle:
		value = glm::radians(glm::gaussRand(treeData->m_parameters.m_branchingAngleMean, treeData->m_parameters.m_branchingAngleVariance));
		break;
	case GrowthParameterType::ApicalAngle:
		value = glm::radians(glm::gaussRand(treeData->m_parameters.m_apicalAngleMean, treeData->m_parameters.m_apicalAngleVariance));
		break;
	case GrowthParameterType::RollAngle:
		value = glm::radians(glm::gaussRand(treeData->m_parameters.m_rollAngleMean, treeData->m_parameters.m_rollAngleVariance));
		break;
	case GrowthParameterType::LateralBudPerNode:
		value = treeData->m_parameters.m_lateralBudPerNode;
		break;
	case GrowthParameterType::ResourceWeightApical:
		value = treeData->m_parameters.m_resourceWeightApical * (1.0f + glm::gaussRand(0.0f, treeData->m_parameters.m_resourceWeightVariance));
		break;
	case GrowthParameterType::ResourceWeightLateral:
		value = 1.0f + glm::gaussRand(0.0f, treeData->m_parameters.m_resourceWeightVariance);
		break;
	case GrowthParameterType::AvoidanceAngle:
		value = treeData->m_parameters.m_avoidanceAngle;
		break;
	}
	return value;
}

void TreeManager::PruneTrees(PlantManager& manager, std::vector<Volume*>& obstacles)
{
	auto& treeManager = GetInstance();
	treeManager.m_voxelSpaceModule.Clear();
	EntityManager::ForEach<GlobalTransform, InternodeInfo>(JobManager::PrimaryWorkers(), manager.m_internodeQuery, [&](int index, Entity internode, GlobalTransform& globalTransform, InternodeInfo& internodeInfo)
		{
			treeManager.m_voxelSpaceModule.Push(globalTransform.GetPosition(), internodeInfo.m_plant, internode);
		}
	);

	std::vector<float> distanceLimits;
	std::vector<float> randomCutOffs;
	std::vector<float> randomCutOffAgeFactors;
	std::vector<float> randomCutOffMaxes;
	std::vector<float> avoidanceAngles;
	std::vector<float> internodeLengths;
	std::vector<RadialBoundingVolume*> rbvs;
	distanceLimits.resize(manager.m_plants.size());
	randomCutOffs.resize(manager.m_plants.size());
	randomCutOffAgeFactors.resize(manager.m_plants.size());
	randomCutOffMaxes.resize(manager.m_plants.size());
	avoidanceAngles.resize(manager.m_plants.size());
	internodeLengths.resize(manager.m_plants.size());
	rbvs.resize(manager.m_plants.size());
	for (int i = 0; i < manager.m_plants.size(); i++)
	{
		if (manager.m_plants[i].HasPrivateComponent<TreeData>()) {
			auto& treeData = manager.m_plants[i].GetPrivateComponent<TreeData>();
			distanceLimits[i] = 0;
			EntityManager::ForEachChild(manager.m_plants[i], [&](Entity child) { if (child.HasComponentData<InternodeInfo>()) distanceLimits[i] = child.GetComponentData<InternodeStatistics>().m_longestDistanceToAnyEndNode; });
			distanceLimits[i] *= treeData->m_parameters.m_lowBranchCutOff;
			randomCutOffs[i] = treeData->m_parameters.m_randomCutOff;
			randomCutOffAgeFactors[i] = treeData->m_parameters.m_randomCutOffAgeFactor;
			randomCutOffMaxes[i] = treeData->m_parameters.m_randomCutOffMax;
			avoidanceAngles[i] = treeData->m_parameters.m_avoidanceAngle;
			internodeLengths[i] = treeData->m_parameters.m_internodeLengthBase;
			rbvs[i] = GetRbv(manager.m_plants[i]).GetPrivateComponent<RadialBoundingVolume>().get();
		}
	}
	std::vector<Entity> cutOff;
	std::mutex mutex;
	EntityManager::ForEach<GlobalTransform, InternodeInfo, Illumination, InternodeStatistics, InternodeGrowth>(JobManager::PrimaryWorkers(), manager.m_internodeQuery, [&](int index, Entity internode, GlobalTransform& globalTransform, InternodeInfo& internodeInfo, Illumination& illumination, InternodeStatistics& internodeStatistics, InternodeGrowth& internodeGrowth)
		{
			if (internodeInfo.m_plantType != PlantType::GeneralTree) return;
			int targetIndex = 0;
			const auto position = globalTransform.GetPosition();
			for (auto* obstacle : obstacles)
			{
				if (obstacle->InVolume(position))
				{
					std::lock_guard lock(mutex);
					cutOff.push_back(internode);
					return;
				}
			}
			for (int i = 0; i < manager.m_plants.size(); i++)
			{
				if (manager.m_plants[i] == internodeInfo.m_plant)
				{
					targetIndex = i;
					break;
				}
			}
			if (internodeInfo.m_order != 1 && internodeGrowth.m_distanceToRoot < distanceLimits[targetIndex]) {
				std::lock_guard lock(mutex);
				cutOff.push_back(internode);
				return;
			}
			if (!rbvs[targetIndex]->InVolume(position)) {
				std::lock_guard lock(mutex);
				cutOff.push_back(internode);
				return;
			}
			//Below are pruning process which only for end nodes.
			if (!internodeStatistics.m_isEndNode) return;
			const glm::vec3 direction = globalTransform.GetRotation() * glm::vec3(0, 0, -1);
			const float angle = avoidanceAngles[targetIndex];
			if (angle > 0 && treeManager.m_voxelSpaceModule.HasObstacleConeSameOwner(angle, globalTransform.GetPosition() - direction * internodeLengths[targetIndex], direction, internodeInfo.m_plant, internode, EntityManager::GetParent(internode), internodeLengths[targetIndex]))
			{
				std::lock_guard lock(mutex);
				cutOff.push_back(internode);
				return;
			}
			if (treeManager.m_voxelSpaceModule.HasNeighborFromDifferentOwner(globalTransform.GetPosition(), internodeInfo.m_plant, treeManager.m_crownShynessDiameter))
			{
				std::lock_guard lock(mutex);
				cutOff.push_back(internode);
				return;
			}
			const float randomCutOffProb = glm::min(manager.m_deltaTime * randomCutOffMaxes[targetIndex], manager.m_deltaTime * randomCutOffs[targetIndex] + (manager.m_globalTime - internodeInfo.m_startGlobalTime) * randomCutOffAgeFactors[targetIndex]);
			if (glm::linearRand(0.0f, 1.0f) < randomCutOffProb) {
				std::lock_guard lock(mutex);
				cutOff.push_back(internode);
				return;
			}
		}, false
		);
	for (const auto& i : cutOff)
		EntityManager::DeleteEntity(i);
}

void TreeManager::UpdateTreesMetaData(PlantManager& manager)
{
	EntityManager::ForEach<PlantInfo, GlobalTransform>(JobManager::PrimaryWorkers(), manager.m_plantQuery, [](int i, Entity tree, PlantInfo& plantInfo, GlobalTransform& globalTransform)
		{
			if (plantInfo.m_plantType != PlantType::GeneralTree) return;
			const Entity rootInternode = GetRootInternode(tree);
			if (rootInternode.IsValid()) {
				auto rootInternodeGrowth = rootInternode.GetComponentData<InternodeGrowth>();
				rootInternodeGrowth.m_desiredGlobalPosition = glm::vec3(0.0f);
				rootInternodeGrowth.m_desiredGlobalRotation = globalTransform.GetRotation() * rootInternodeGrowth.m_desiredLocalRotation;
				rootInternode.SetComponentData(rootInternodeGrowth);
				auto& treeData = tree.GetPrivateComponent<TreeData>();
				UpdateDistances(rootInternode, treeData);
				UpdateLevels(rootInternode, treeData);
			}
		}, false
		);
}

void TreeManager::UpdateDistances(const Entity& internode, std::unique_ptr<TreeData>& treeData)
{
	Entity currentInternode = internode;
	auto currentInternodeInfo = internode.GetComponentData<InternodeInfo>();
	auto currentInternodeGrowth = internode.GetComponentData<InternodeGrowth>();
	auto currentInternodeStatistics = internode.GetComponentData<InternodeStatistics>();
#pragma region Single child chain from root to branch
	while (EntityManager::GetChildrenAmount(currentInternode) == 1)
	{
#pragma region Retrive child status
		Entity child = EntityManager::GetChildren(currentInternode)[0];
		auto childInternodeGrowth = child.GetComponentData<InternodeGrowth>();
		auto childInternodeStatistics = child.GetComponentData<InternodeStatistics>();
		auto childInternodeInfo = child.GetComponentData<InternodeInfo>();
#pragma endregion
#pragma region Update child status
		childInternodeGrowth.m_inhibitor = 0;
		childInternodeStatistics.m_distanceToBranchStart = currentInternodeStatistics.m_distanceToBranchStart + 1;
		childInternodeGrowth.m_desiredGlobalRotation = currentInternodeGrowth.m_desiredGlobalRotation * childInternodeGrowth.m_desiredLocalRotation;
		childInternodeGrowth.m_desiredGlobalPosition = currentInternodeGrowth.m_desiredGlobalPosition + treeData->m_parameters.m_internodeLengthBase * (currentInternodeGrowth.m_desiredGlobalRotation * glm::vec3(0, 0, -1));
#pragma endregion
#pragma region Apply child status
		child.SetComponentData(childInternodeStatistics);
		child.SetComponentData(childInternodeGrowth);
		child.SetComponentData(childInternodeInfo);
#pragma endregion
#pragma region Retarget current internode
		currentInternode = child;
		currentInternodeInfo = childInternodeInfo;
		currentInternodeGrowth = childInternodeGrowth;
		currentInternodeStatistics = childInternodeStatistics;
#pragma endregion
	}
#pragma region Reset current status
	currentInternodeGrowth.m_inhibitor = 0;
	currentInternodeStatistics.m_totalLength = 0;
	currentInternodeStatistics.m_longestDistanceToAnyEndNode = 0;
	currentInternodeStatistics.m_maxChildOrder = 0;
	currentInternodeStatistics.m_isEndNode = false;
	currentInternodeStatistics.m_childrenEndNodeAmount = 0;
	currentInternodeGrowth.m_thickness = 0;
	currentInternodeGrowth.m_childrenTotalTorque = glm::vec3(0.0f);
	currentInternodeGrowth.m_MassOfChildren = 0.0f;
#pragma endregion
#pragma endregion
	if (EntityManager::GetChildrenAmount(currentInternode) != 0)
	{
		float maxThickness = 0;
		EntityManager::ForEachChild(currentInternode, [&](Entity child)
			{
#pragma region From root to end
#pragma region Retrive child status
				auto childInternodeInfo = child.GetComponentData<InternodeInfo>();
				auto childInternodeGrowth = child.GetComponentData<InternodeGrowth>();
				auto childInternodeStatistics = child.GetComponentData<InternodeStatistics>();
#pragma endregion
#pragma region Update child status
				childInternodeStatistics.m_distanceToBranchStart =
					(currentInternodeInfo.m_order == childInternodeInfo.m_order ? currentInternodeStatistics.m_distanceToBranchStart : 0)
					+ 1;
				childInternodeGrowth.m_desiredGlobalRotation = currentInternodeGrowth.m_desiredGlobalRotation * childInternodeGrowth.m_desiredLocalRotation;
				childInternodeGrowth.m_desiredGlobalPosition = currentInternodeGrowth.m_desiredGlobalPosition + treeData->m_parameters.m_internodeLengthBase * (currentInternodeGrowth.m_desiredGlobalRotation * glm::vec3(0, 0, -1));

#pragma endregion
#pragma region Apply child status
				child.SetComponentData(childInternodeStatistics);
				child.SetComponentData(childInternodeGrowth);
				child.SetComponentData(childInternodeInfo);
#pragma endregion
#pragma endregion
				UpdateDistances(child, treeData);
#pragma region From end to root
#pragma region Retrive child status
				childInternodeInfo = child.GetComponentData<InternodeInfo>();
				childInternodeGrowth = child.GetComponentData<InternodeGrowth>();
				childInternodeStatistics = child.GetComponentData<InternodeStatistics>();
#pragma endregion
#pragma region Update self status
				auto& childInternodeData = child.GetPrivateComponent<InternodeData>();
				for (const auto& bud : childInternodeData->m_buds) if (bud.m_active && bud.m_isApical) currentInternodeGrowth.m_inhibitor += treeData->m_parameters.m_inhibitorBase;
				currentInternodeGrowth.m_inhibitor += childInternodeGrowth.m_inhibitor * childInternodeGrowth.m_inhibitorTransmitFactor;
				currentInternodeStatistics.m_childrenEndNodeAmount += childInternodeStatistics.m_childrenEndNodeAmount;
				if (currentInternodeStatistics.m_maxChildOrder < childInternodeStatistics.m_maxChildOrder) currentInternodeStatistics.m_maxChildOrder = childInternodeStatistics.m_maxChildOrder;
				currentInternodeStatistics.m_totalLength += childInternodeStatistics.m_totalLength + 1;
				const int tempDistanceToEndNode = childInternodeStatistics.m_longestDistanceToAnyEndNode + 1;
				if (currentInternodeStatistics.m_longestDistanceToAnyEndNode < tempDistanceToEndNode) currentInternodeStatistics.m_longestDistanceToAnyEndNode = tempDistanceToEndNode;
				currentInternodeGrowth.m_thickness += glm::pow(childInternodeGrowth.m_thickness, 2.0f);
				currentInternodeGrowth.m_MassOfChildren += childInternodeGrowth.m_MassOfChildren + childInternodeGrowth.m_thickness;
				currentInternodeGrowth.m_childrenTotalTorque += childInternodeGrowth.m_childrenTotalTorque +
					childInternodeGrowth.m_desiredGlobalPosition * childInternodeGrowth.m_thickness;

				if (childInternodeGrowth.m_thickness > maxThickness) {
					maxThickness = childInternodeGrowth.m_thickness;
					currentInternodeGrowth.m_thickestChild = child;
				}
#pragma endregion
#pragma endregion
			}
		);
		currentInternodeGrowth.m_thickness = glm::pow(currentInternodeGrowth.m_thickness, 0.5f) * treeData->m_parameters.m_thicknessControlFactor;
		currentInternodeGrowth.m_childMeanPosition = currentInternodeGrowth.m_childrenTotalTorque / currentInternodeGrowth.m_MassOfChildren;
		const float strength = currentInternodeGrowth.m_MassOfChildren * glm::distance(glm::vec2(currentInternodeGrowth.m_childMeanPosition.x, currentInternodeGrowth.m_childMeanPosition.z), glm::vec2(currentInternodeGrowth.m_desiredGlobalPosition.x, currentInternodeGrowth.m_desiredGlobalPosition.z));
		currentInternodeGrowth.m_sagging = glm::min(treeData->m_parameters.m_gravityBendingMax
			, strength * treeData->m_parameters.m_gravityBendingFactor / glm::pow(currentInternodeGrowth.m_thickness / treeData->m_parameters.m_endNodeThickness, treeData->m_parameters.m_gravityBendingThicknessFactor));
	}
	else {
#pragma region Update self status (end node)
		currentInternodeStatistics.m_childrenEndNodeAmount = 1;
		currentInternodeStatistics.m_isEndNode = true;
		currentInternodeStatistics.m_maxChildOrder = currentInternodeInfo.m_order;
		currentInternodeStatistics.m_totalLength = 0;
		currentInternodeStatistics.m_longestDistanceToAnyEndNode = 0;
		currentInternodeGrowth.m_thickness = treeData->m_parameters.m_endNodeThickness;
		currentInternodeGrowth.m_thickestChild = Entity();
#pragma endregion
	}
#pragma region From end to root
	currentInternodeStatistics.m_distanceToBranchEnd = 0;
	while (currentInternode != internode)
	{
#pragma region Apply current status
		currentInternode.SetComponentData(currentInternodeInfo);
		currentInternode.SetComponentData(currentInternodeGrowth);
		currentInternode.SetComponentData(currentInternodeStatistics);
#pragma endregion
#pragma region Retarget to parent
		auto& childInternodeData = currentInternode.GetPrivateComponent<InternodeData>();
		Entity child = currentInternode;
		currentInternode = EntityManager::GetParent(currentInternode);
		auto childInternodeInfo = currentInternodeInfo;
		auto childInternodeGrowth = currentInternodeGrowth;
		auto childInternodeStatistics = currentInternodeStatistics;
		currentInternodeInfo = currentInternode.GetComponentData<InternodeInfo>();
		currentInternodeGrowth = currentInternode.GetComponentData<InternodeGrowth>();
		currentInternodeStatistics = currentInternode.GetComponentData<InternodeStatistics>();
#pragma endregion
#pragma region Reset current status
		currentInternodeGrowth.m_inhibitor = 0;
		currentInternodeStatistics.m_totalLength = 0;
		currentInternodeStatistics.m_longestDistanceToAnyEndNode = 0;
		currentInternodeStatistics.m_maxChildOrder = 0;
		currentInternodeStatistics.m_isEndNode = false;
		currentInternodeStatistics.m_childrenEndNodeAmount = 0;
		currentInternodeGrowth.m_thickness = 0;
		currentInternodeGrowth.m_childrenTotalTorque = glm::vec3(0.0f);
		currentInternodeGrowth.m_MassOfChildren = 0.0f;
#pragma endregion
#pragma region Update self status
		for (const auto& bud : childInternodeData->m_buds) if (bud.m_active && bud.m_isApical) currentInternodeGrowth.m_inhibitor += treeData->m_parameters.m_inhibitorBase;
		currentInternodeGrowth.m_inhibitor += childInternodeGrowth.m_inhibitor * childInternodeGrowth.m_inhibitorTransmitFactor;
		currentInternodeStatistics.m_childrenEndNodeAmount = childInternodeStatistics.m_childrenEndNodeAmount;
		currentInternodeGrowth.m_thickness = childInternodeGrowth.m_thickness;
		currentInternodeStatistics.m_maxChildOrder = childInternodeStatistics.m_maxChildOrder;
		currentInternodeStatistics.m_isEndNode = false;
		currentInternodeStatistics.m_distanceToBranchEnd = childInternodeStatistics.m_distanceToBranchEnd + 1;
		currentInternodeStatistics.m_totalLength = childInternodeStatistics.m_totalLength + 1;
		currentInternodeStatistics.m_longestDistanceToAnyEndNode = childInternodeStatistics.m_longestDistanceToAnyEndNode + 1;

		currentInternodeGrowth.m_MassOfChildren += childInternodeGrowth.m_MassOfChildren + childInternodeGrowth.m_thickness;
		currentInternodeGrowth.m_childrenTotalTorque += childInternodeGrowth.m_childrenTotalTorque +
			childInternodeGrowth.m_desiredGlobalPosition * childInternodeGrowth.m_thickness;
		currentInternodeGrowth.m_childMeanPosition = currentInternodeGrowth.m_childrenTotalTorque / currentInternodeGrowth.m_MassOfChildren;
		const float strength = currentInternodeGrowth.m_MassOfChildren * glm::distance(glm::vec2(currentInternodeGrowth.m_childMeanPosition.x, currentInternodeGrowth.m_childMeanPosition.z), glm::vec2(currentInternodeGrowth.m_desiredGlobalPosition.x, currentInternodeGrowth.m_desiredGlobalPosition.z));
		currentInternodeGrowth.m_sagging = strength * treeData->m_parameters.m_gravityBendingFactor / glm::pow(currentInternodeGrowth.m_thickness / treeData->m_parameters.m_endNodeThickness, treeData->m_parameters.m_gravityBendingThicknessFactor);
		currentInternodeGrowth.m_thickestChild = child;
#pragma endregion
	}
#pragma endregion
#pragma region Apply self status
	currentInternode.SetComponentData(currentInternodeInfo);
	currentInternode.SetComponentData(currentInternodeGrowth);
	currentInternode.SetComponentData(currentInternodeStatistics);
#pragma endregion
}

void TreeManager::UpdateLevels(const Entity& internode, std::unique_ptr<TreeData>& treeData)
{
	auto currentInternode = internode;
	auto currentInternodeInfo = internode.GetComponentData<InternodeInfo>();
	auto currentInternodeGlobalTransform = internode.GetComponentData<GlobalTransform>();
#pragma region Single child chain from root to branch
	while (EntityManager::GetChildrenAmount(currentInternode) == 1)
	{
#pragma region Retrive child status
		Entity child = EntityManager::GetChildren(currentInternode)[0];
		auto childInternodeInfo = child.GetComponentData<InternodeInfo>();
		auto childInternodeGrowth = child.GetComponentData<InternodeGrowth>();
#pragma endregion
#pragma region Update child status
		childInternodeInfo.m_level = currentInternodeInfo.m_level;
#pragma region Gravity bending.
		Transform childInternodeTransform;
		GlobalTransform childInternodeGlobalTransform;
		glm::quat globalRotation = currentInternodeGlobalTransform.GetRotation() * childInternodeGrowth.m_desiredLocalRotation;
		glm::vec3 front = globalRotation * glm::vec3(0, 0, -1);
		glm::vec3 up = globalRotation * glm::vec3(0, 1, 0);
		PlantManager::ApplyTropism(glm::vec3(0, -1, 0), childInternodeGrowth.m_sagging, front, up);
		globalRotation = glm::quatLookAt(front, up);
		const glm::vec3 globalPosition = currentInternodeGlobalTransform.GetPosition() + front * treeData->m_parameters.m_internodeLengthBase;
		childInternodeGlobalTransform.SetValue(globalPosition, globalRotation, glm::vec3(1.0f));
		childInternodeTransform.m_value = glm::inverse(currentInternodeGlobalTransform.m_value) * childInternodeGlobalTransform.m_value;
#pragma endregion
#pragma endregion
#pragma region Apply child status
		child.SetComponentData(childInternodeInfo);
		child.SetComponentData(childInternodeTransform);
		child.SetComponentData(childInternodeGlobalTransform);
		child.SetComponentData(childInternodeGrowth);
#pragma endregion
#pragma region Retarget current internode
		currentInternode = child;
		currentInternodeInfo = childInternodeInfo;
		currentInternodeGlobalTransform = childInternodeGlobalTransform;
#pragma endregion
	}
	auto currentInternodeStatistics = currentInternode.GetComponentData<InternodeStatistics>();
#pragma endregion
	if (EntityManager::GetChildrenAmount(currentInternode) != 0)
	{
#pragma region Select max child
		float maxChildLength = 0;
		int minChildOrder = 9999;
		Entity maxChild = Entity();
		EntityManager::ForEachChild(currentInternode, [&](Entity child)
			{
				const auto childInternodeStatistics = child.GetComponentData<InternodeStatistics>();
				const auto childInternodeInfo = child.GetComponentData<InternodeInfo>();
				if (maxChildLength <= childInternodeStatistics.m_totalLength && childInternodeInfo.m_order < minChildOrder)
				{
					minChildOrder = childInternodeInfo.m_order;
					maxChildLength = childInternodeStatistics.m_totalLength;
					maxChild = child;
				}
			}
		);
#pragma endregion
#pragma region Apply level
		EntityManager::ForEachChild(currentInternode, [&](Entity child)
			{
				auto childInternodeInfo = child.GetComponentData<InternodeInfo>();
				auto childInternodeStatistics = child.GetComponentData<InternodeStatistics>();
				auto childInternodeGrowth = child.GetComponentData<InternodeGrowth>();

				if (child == maxChild)
				{
					childInternodeStatistics.m_isMaxChild = true;
					childInternodeInfo.m_level = currentInternodeInfo.m_level;
				}
				else
				{
					childInternodeStatistics.m_isMaxChild = false;
					childInternodeInfo.m_level = currentInternodeInfo.m_level + 1;
				}
#pragma region Gravity bending.
				Transform childInternodeTransform;
				GlobalTransform childInternodeGlobalTransform;
				glm::quat globalRotation = currentInternodeGlobalTransform.GetRotation() * childInternodeGrowth.m_desiredLocalRotation;
				glm::vec3 front = globalRotation * glm::vec3(0, 0, -1);
				glm::vec3 up = globalRotation * glm::vec3(0, 1, 0);
				PlantManager::ApplyTropism(childInternodeGrowth.m_desiredGlobalPosition - childInternodeGrowth.m_childMeanPosition, childInternodeGrowth.m_sagging, front, up);
				globalRotation = glm::quatLookAt(front, up);
				const glm::vec3 globalPosition = currentInternodeGlobalTransform.GetPosition() + front * treeData->m_parameters.m_internodeLengthBase;
				childInternodeGlobalTransform.SetValue(globalPosition, globalRotation, glm::vec3(1.0f));
				childInternodeTransform.m_value = glm::inverse(currentInternodeGlobalTransform.m_value) * childInternodeGlobalTransform.m_value;

#pragma endregion
#pragma region Apply child status
				child.SetComponentData(childInternodeTransform);
				child.SetComponentData(childInternodeGlobalTransform);
				child.SetComponentData(childInternodeStatistics);
				child.SetComponentData(childInternodeInfo);
				child.SetComponentData(childInternodeGrowth);
#pragma endregion
				UpdateLevels(child, treeData);
				childInternodeStatistics = child.GetComponentData<InternodeStatistics>();
				if (childInternodeStatistics.m_maxChildLevel > currentInternodeStatistics.m_maxChildLevel) currentInternodeStatistics.m_maxChildLevel = childInternodeStatistics.m_maxChildLevel;
			}
		);
#pragma endregion
	}
	else
	{
		currentInternodeStatistics.m_maxChildLevel = currentInternodeInfo.m_level;
	}
	while (currentInternode != internode)
	{
#pragma region Apply current status
		currentInternode.SetComponentData(currentInternodeStatistics);
#pragma endregion
#pragma region Retarget to parent
		currentInternode = EntityManager::GetParent(currentInternode);
		const auto childInternodeStatistics = currentInternodeStatistics;
		currentInternodeStatistics = currentInternode.GetComponentData<InternodeStatistics>();
#pragma endregion
#pragma region Update self status
		currentInternodeStatistics.m_maxChildLevel = childInternodeStatistics.m_maxChildLevel;
#pragma endregion
	}
#pragma region Apply self status
	currentInternode.SetComponentData(currentInternodeStatistics);
#pragma endregion
}

void TreeManager::ResetTimeForTree(const float& value)
{
	auto& manager = PlantManager::GetInstance();
	if (value < 0 || value >= manager.m_globalTime) return;
	manager.m_globalTime = value;
	std::vector<Entity> trees;
	manager.m_plantQuery.ToEntityArray(trees);
	for (const auto& tree : trees)
	{
		auto plantInfo = tree.GetComponentData<PlantInfo>();
		if (plantInfo.m_startTime > value) {
			EntityManager::DeleteEntity(tree);
			continue;
		}
		plantInfo.m_age = value - plantInfo.m_startTime;
		tree.SetComponentData(plantInfo);
		Entity rootInternode = GetRootInternode(tree);
		if (rootInternode.IsValid()) {
			ResetTimeForTree(rootInternode, value);
		}
	}
	EntityManager::ForEach<InternodeInfo>(JobManager::PrimaryWorkers(), manager.m_internodeQuery, [value](int i, Entity internode, InternodeInfo& internodeInfo)
		{
			auto& childInternodeData = internode.GetPrivateComponent<InternodeData>();
			for (auto& bud : childInternodeData->m_buds)
			{
				if (!bud.m_active && bud.m_deathGlobalTime > value)
				{
					bud.m_active = true;
					bud.m_deathGlobalTime = -1;
				}
				bud.m_enoughForGrowth = false;
				bud.m_currentResource = ResourceParcel();
				for (auto it = bud.m_resourceLog.begin(); it != bud.m_resourceLog.end(); ++it)
				{
					if (it->m_globalTime > value)
					{
						bud.m_resourceLog.erase(it, bud.m_resourceLog.end());
						break;
					}
				}
				for (const auto& parcel : bud.m_resourceLog) bud.m_currentResource += parcel;
				if (bud.m_currentResource.IsEnough()) bud.m_enoughForGrowth = true;
			}
		}, false
		);
	UpdateTreesMetaData(manager);
}

void TreeManager::ResetTimeForTree(const Entity& internode, const float& globalTime)
{
	Entity currentInternode = internode;
	while (EntityManager::GetChildrenAmount(currentInternode) == 1)
	{
		Entity child = EntityManager::GetChildren(currentInternode)[0];
		const auto childInternodeInfo = child.GetComponentData<InternodeInfo>();
		if (childInternodeInfo.m_startGlobalTime > globalTime) {
			EntityManager::DeleteEntity(child);
			return;
		}
		currentInternode = child;
	}

	if (EntityManager::GetChildrenAmount(currentInternode) != 0)
	{
		std::vector<Entity> childrenToDelete;
		EntityManager::ForEachChild(currentInternode, [globalTime, &childrenToDelete](Entity child)
			{
				const auto childInternodeInfo = child.GetComponentData<InternodeInfo>();
				if (childInternodeInfo.m_startGlobalTime > globalTime) {
					childrenToDelete.push_back(child);
					return;
				}
				ResetTimeForTree(child, globalTime);
			}
		);
		for (const auto& child : childrenToDelete) EntityManager::DeleteEntity(child);
	}
}

void TreeManager::DistributeResourcesForTree(PlantManager& manager,
	std::vector<ResourceParcel>& totalNutrients)
{
	auto& plants = manager.m_plants;
	std::vector<float> divisors;
	std::vector<float> apicalControlLevelFactors;
	std::vector<float> resourceAllocationDistFactors;
	std::vector<float> apicalIlluminationRequirements;
	std::vector<float> lateralIlluminationRequirements;
	std::vector<float> requirementMaximums;
	std::vector<glm::vec3> treePositions;
	std::vector<float> heightResourceBase;
	std::vector<float> heightResourceFactor;
	std::vector<float> heightResourceFactorMin;
	divisors.resize(plants.size());
	apicalControlLevelFactors.resize(plants.size());
	resourceAllocationDistFactors.resize(plants.size());
	apicalIlluminationRequirements.resize(plants.size());
	lateralIlluminationRequirements.resize(plants.size());
	requirementMaximums.resize(plants.size());
	treePositions.resize(plants.size());
	heightResourceBase.resize(plants.size());
	heightResourceFactor.resize(plants.size());
	heightResourceFactorMin.resize(plants.size());
	for (int i = 0; i < plants.size(); i++)
	{
		if (plants[i].HasPrivateComponent<TreeData>()) {
			auto& treeData = plants[i].GetPrivateComponent<TreeData>();
			divisors[i] = 0;
			apicalControlLevelFactors[i] = treeData->m_parameters.m_apicalControlLevelFactor;
			apicalIlluminationRequirements[i] = treeData->m_parameters.m_apicalIlluminationRequirement;
			lateralIlluminationRequirements[i] = treeData->m_parameters.m_lateralIlluminationRequirement;
			requirementMaximums[i] = 0;
			treePositions[i] = plants[i].GetComponentData<GlobalTransform>().GetPosition();
			heightResourceBase[i] = treeData->m_parameters.m_heightResourceHeightDecreaseBase;
			heightResourceFactor[i] = treeData->m_parameters.m_heightResourceHeightDecreaseFactor;
			heightResourceFactorMin[i] = treeData->m_parameters.m_heightResourceHeightDecreaseMin;
		}
	}
	std::mutex maximumLock;
	EntityManager::ForEach<GlobalTransform, InternodeInfo, InternodeGrowth, InternodeStatistics>(JobManager::PrimaryWorkers(), manager.m_internodeQuery,
		[&](int index, Entity internode, GlobalTransform& globalTransform, InternodeInfo& internodeInfo, InternodeGrowth& internodeGrowth, InternodeStatistics& internodeStatistics)
		{
			if (internodeInfo.m_plantType != PlantType::GeneralTree) return;
			for (int i = 0; i < plants.size(); i++)
			{
				if (plants[i] == internodeInfo.m_plant)
				{
					const float internodeRequirement =
						glm::max(heightResourceFactorMin[i], 1.0f - glm::pow(globalTransform.GetPosition().y - treePositions[i].y, heightResourceFactor[i]) * heightResourceBase[i])
						* glm::max(0.0f, 1.0f - internodeGrowth.m_inhibitor)
						* glm::pow(apicalControlLevelFactors[i], static_cast<float>(internodeInfo.m_level))
						;
					auto& internodeData = internode.GetPrivateComponent<InternodeData>();
					float budsRequirement = 0;
					for (const auto& bud : internodeData->m_buds)
					{
						if (bud.m_active && !bud.m_enoughForGrowth) {
							budsRequirement += bud.m_resourceWeight;
							std::lock_guard<std::mutex> lock(maximumLock);
							const float budRequirement = internodeRequirement * bud.m_resourceWeight;
							if (budRequirement > requirementMaximums[i])
							{
								requirementMaximums[i] = budRequirement;
							}
						}
					}
					divisors[i] += budsRequirement * internodeRequirement;
					break;
				}
			}
		}, false
		);
	const auto globalTime = manager.m_globalTime;
	EntityManager::ForEach<GlobalTransform, Illumination, InternodeInfo, InternodeGrowth, InternodeStatistics>(JobManager::PrimaryWorkers(), manager.m_internodeQuery,
		[=, &manager](int index, Entity internode, GlobalTransform& globalTransform, Illumination& illumination, InternodeInfo& internodeInfo, InternodeGrowth& internodeGrowth, InternodeStatistics& internodeStatistics)
		{
			if (internodeInfo.m_plantType != PlantType::GeneralTree) return;
			for (int i = 0; i < plants.size(); i++)
			{
				if (plants[i] == internodeInfo.m_plant)
				{
					auto& internodeData = internode.GetPrivateComponent<InternodeData>();
					float budsRequirement = 0;
					for (const auto& bud : internodeData->m_buds)
					{
						if (bud.m_active && !bud.m_enoughForGrowth) budsRequirement += bud.m_resourceWeight;
					}
					const float internodeRequirement =
						glm::max(heightResourceFactorMin[i], 1.0f - glm::pow(globalTransform.GetPosition().y - treePositions[i].y, heightResourceFactor[i]) * heightResourceBase[i])
						* glm::max(0.0f, 1.0f - internodeGrowth.m_inhibitor)
						* glm::pow(apicalControlLevelFactors[i], static_cast<float>(internodeInfo.m_level))
						;
					const float internodeNutrient = glm::min(totalNutrients[i].m_nutrient / (divisors[i] / requirementMaximums[i]), manager.m_deltaTime) * budsRequirement * internodeRequirement / requirementMaximums[i];
					const float internodeCarbon = illumination.m_currentIntensity * manager.m_deltaTime;
					for (auto& bud : internodeData->m_buds)
					{
						if (bud.m_active && !bud.m_enoughForGrowth) {
							ResourceParcel resourceParcel = ResourceParcel(
								internodeNutrient / budsRequirement * bud.m_resourceWeight,
								internodeCarbon / (bud.m_isApical ? apicalIlluminationRequirements[i] : lateralIlluminationRequirements[i]));
							resourceParcel.m_globalTime = globalTime;
							bud.m_currentResource += resourceParcel;
							bud.m_resourceLog.push_back(resourceParcel);
							if (bud.m_currentResource.IsEnough()) bud.m_enoughForGrowth = true;
						}
					}
					break;
				}
			}
		}, false
		);
}

void TreeManager::Init()
{
	auto& manager = GetInstance();
	manager.m_voxelSpaceModule.Reset();


	manager.m_colorMapSegmentAmount = 3;
	manager.m_colorMapValues.resize(manager.m_colorMapSegmentAmount);
	manager.m_colorMapColors.resize(manager.m_colorMapSegmentAmount);
	for (int i = 0; i < manager.m_colorMapSegmentAmount; i++)
	{
		manager.m_colorMapValues[i] = i;
		manager.m_colorMapColors[i] = glm::linearRand(glm::vec3(0.0f), glm::vec3(1.0f));
	}

#pragma region Internode camera
	manager.m_internodeDebuggingCamera = std::make_unique<CameraComponent>();
	manager.m_internodeDebuggingCamera->m_drawSkyBox = false;
	manager.m_internodeDebuggingCamera->m_clearColor = glm::vec3(0.1f);
	manager.m_internodeDebuggingCamera->m_skyBox = DefaultResources::Textures::DefaultSkybox;

#pragma endregion

	manager.m_leavesArchetype = EntityManager::CreateEntityArchetype("Tree Leaves", TreeLeavesTag());
	manager.m_rbvArchetype = EntityManager::CreateEntityArchetype("RBV", RbvTag());

	auto& plantManager = PlantManager::GetInstance();
#pragma region Materials
	for (int i = 0; i < 64; i++)
	{
		manager.m_randomColors.emplace_back(glm::linearRand(0.0f, 1.0f), glm::linearRand(0.0f, 1.0f), glm::linearRand(0.0f, 1.0f));
	}
	manager.m_defaultRayTracingBranchAlbedoTexture = ResourceManager::LoadTexture(false, FileIO::GetAssetFolderPath() + "Textures/BarkMaterial/Bark_Pine_baseColor.jpg", TextureType::Albedo);
	manager.m_defaultRayTracingBranchNormalTexture = ResourceManager::LoadTexture(false, FileIO::GetAssetFolderPath() + "Textures/BarkMaterial/Bark_Pine_normal.jpg", TextureType::Normal);
	manager.m_defaultBranchAlbedoTexture = ResourceManager::LoadTexture(false, FileIO::GetAssetFolderPath() + "Textures/BarkMaterial/Bark_Pine_baseColor.jpg", TextureType::Albedo);
	manager.m_defaultBranchNormalTexture = ResourceManager::LoadTexture(false, FileIO::GetAssetFolderPath() + "Textures/BarkMaterial/Bark_Pine_normal.jpg", TextureType::Normal);
#pragma endregion
#pragma region General tree growth
	plantManager.m_plantMeshGenerators.insert_or_assign(PlantType::GeneralTree, [](PlantManager& manager)
		{
			GenerateMeshForTree(manager);
		});

	plantManager.m_plantFoliageGenerators.insert_or_assign(PlantType::GeneralTree, [](PlantManager& manager)
		{
			GenerateLeavesForTree(manager);
		});

	plantManager.m_plantResourceAllocators.insert_or_assign(PlantType::GeneralTree, [](PlantManager& manager, std::vector<ResourceParcel>& resources)
		{
			DistributeResourcesForTree(manager, resources);
		}
	);

	plantManager.m_plantGrowthModels.insert_or_assign(PlantType::GeneralTree, [](PlantManager& manager, std::vector<InternodeCandidate>& candidates)
		{
			FormCandidates(manager, candidates);
		}
	);

	plantManager.m_plantInternodePruners.insert_or_assign(PlantType::GeneralTree, [](PlantManager& manager, std::vector<Volume*>& obstacles)
		{
			PruneTrees(manager, obstacles);
		}
	);

	plantManager.m_plantMetaDataCalculators.insert_or_assign(PlantType::GeneralTree, [](PlantManager& manager)
		{
			UpdateTreesMetaData(manager);
		}
	);
#pragma endregion
	EditorManager::RegisterPrivateComponentMenu<CubeVolume>([](Entity owner)
		{
			if (owner.HasPrivateComponent<CubeVolume>()) return;
			if (ImGui::SmallButton("CubeVolume"))
			{
				owner.SetPrivateComponent(std::make_unique<CubeVolume>());
			}
		}
	);
}

void TreeManager::SerializeScene(const std::string& filename)
{
	std::ofstream ofs;
	ofs.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc);
	if (!ofs.is_open())
	{
		Debug::Error("Can't open file!");
		return;
	}
	rapidxml::xml_document<> doc;
	auto* type = doc.allocate_node(rapidxml::node_doctype, 0, "Scene");
	doc.append_node(type);
	auto* scene = doc.allocate_node(rapidxml::node_element, "Scene", "Tree");
	doc.append_node(scene);
	std::vector<Entity> trees;
	PlantManager::GetInstance().m_plantQuery.ToEntityArray(trees);
	for (const auto& plant : trees)
	{
		Serialize(plant, doc, scene);
	}
	ofs << doc;
	ofs.flush();
	ofs.close();
}

void TreeManager::Serialize(const Entity& treeEntity, rapidxml::xml_document<>& doc,
	rapidxml::xml_node<>* sceneNode)
{
	if (treeEntity.GetComponentData<PlantInfo>().m_plantType != PlantType::GeneralTree) return;
	auto* tree = doc.allocate_node(rapidxml::node_element, "Tree", "Textures");
	sceneNode->append_node(tree);

	auto* textures = doc.allocate_node(rapidxml::node_element, "Textures", "Texture");
	tree->append_node(textures);
	auto* barkTex = doc.allocate_node(rapidxml::node_element, "Texture", "");
	barkTex->append_attribute(doc.allocate_attribute("name", "Bark"));
	barkTex->append_attribute(doc.allocate_attribute("path", "Data/Textures/Bark/UlmusLaevis.jpg"));
	auto* leafTex = doc.allocate_node(rapidxml::node_element, "Texture", "");
	leafTex->append_attribute(doc.allocate_attribute("name", "Leaf"));
	leafTex->append_attribute(doc.allocate_attribute("path", "Data/Textures/Leaf/UlmusLaevis"));

	textures->append_node(leafTex);
	textures->append_node(barkTex);

	auto* nodes = doc.allocate_node(rapidxml::node_element, "Nodes", "Node");
	tree->append_node(nodes);

	std::vector<InternodeInfo> internodeInfos;
	std::vector<Entity> internodes;
	PlantManager::GetInstance().m_internodeQuery.ToEntityArray<InternodeInfo>(internodes, [treeEntity](const Entity& entity, const InternodeInfo& internodeInfo)
		{
			return treeEntity == internodeInfo.m_plant;
		}
	);
	PlantManager::GetInstance().m_internodeQuery.ToComponentDataArray<InternodeInfo, InternodeInfo>(internodeInfos, [treeEntity](const InternodeInfo& internodeInfo)
		{
			return treeEntity == internodeInfo.m_plant;
		}
	);
	Entity rootInternode;
	unsigned rootNodeIndex = 0;
	EntityManager::ForEachChild(treeEntity, [&rootNodeIndex, &rootInternode](Entity child)
		{
			if (child.HasComponentData<InternodeInfo>()) {
				rootNodeIndex = child.m_index - 1;
				rootInternode = child;
			}
		});
	rootNodeIndex = 0;
	for (auto& i : internodes)
	{
		auto internodeGrowth = i.GetComponentData<InternodeGrowth>();
		auto internodeInfo = i.GetComponentData<InternodeInfo>();
		auto internodeStatistics = i.GetComponentData<InternodeStatistics>();
		auto* node = doc.allocate_node(rapidxml::node_element, "Node", "Position");
		node->append_attribute(doc.allocate_attribute("id", doc.allocate_string(std::to_string(i.m_index - rootNodeIndex).c_str())));
		node->append_attribute(doc.allocate_attribute("additional", doc.allocate_string(std::to_string(0).c_str())));
		nodes->append_node(node);
		auto globalTransform = i.GetComponentData<GlobalTransform>().m_value;
		auto* position = doc.allocate_node(rapidxml::node_element, "Position");
		position->append_attribute(doc.allocate_attribute("x", doc.allocate_string(std::to_string(globalTransform[3].x).c_str())));
		position->append_attribute(doc.allocate_attribute("y", doc.allocate_string(std::to_string(globalTransform[3].y).c_str())));
		position->append_attribute(doc.allocate_attribute("z", doc.allocate_string(std::to_string(globalTransform[3].z).c_str())));
		node->append_node(position);

		auto* distRoot = doc.allocate_node(rapidxml::node_element, "DistToRoot");
		distRoot->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(internodeGrowth.m_distanceToRoot).c_str())));
		node->append_node(distRoot);

		auto* thickness = doc.allocate_node(rapidxml::node_element, "Thickness");
		thickness->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(internodeGrowth.m_thickness).c_str())));
		node->append_node(thickness);

		auto* maxLength = doc.allocate_node(rapidxml::node_element, "MaxBranchLength");
		maxLength->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(internodeStatistics.m_longestDistanceToAnyEndNode).c_str())));
		node->append_node(maxLength);

		unsigned parentIndex = EntityManager::GetParent(i).m_index;
		float thicknessVal = 0;
		if (parentIndex != treeEntity.m_index)
		{
			thicknessVal = EntityManager::GetParent(i).GetComponentData<InternodeGrowth>().m_thickness;
		}
		else
		{
			auto* root = doc.allocate_node(rapidxml::node_element, "Root");
			root->append_attribute(doc.allocate_attribute("value", "true"));
			node->append_node(root);
		}
		auto* parent = doc.allocate_node(rapidxml::node_element, "Parent");
		parent->append_attribute(doc.allocate_attribute("id", doc.allocate_string(std::to_string(parentIndex - rootNodeIndex).c_str())));
		parent->append_attribute(doc.allocate_attribute("thickness", doc.allocate_string(std::to_string(thicknessVal).c_str())));
		node->append_node(parent);
		if (EntityManager::GetChildrenAmount(i) != 0) {
			auto* children = doc.allocate_node(rapidxml::node_element, "Children", "Child");
			node->append_node(children);

			EntityManager::ForEachChild(i, [children, &doc, rootNodeIndex](Entity child)
				{
					auto* childNode = doc.allocate_node(rapidxml::node_element, "Child");
					childNode->append_attribute(doc.allocate_attribute("id", doc.allocate_string(std::to_string(child.m_index - rootNodeIndex).c_str())));
					children->append_node(childNode);
				}
			);
		}
	}

	auto* chains = doc.allocate_node(rapidxml::node_element, "Chains", "Chain");
	tree->append_node(chains);

	ExportChains(-1, rootInternode, chains, &doc);

	auto* leaves = doc.allocate_node(rapidxml::node_element, "Leaves", "Leaf");
	tree->append_node(leaves);
	int counter = 0;
	for (auto& i : internodes)
	{
		glm::vec3 nodePos = i.GetComponentData<GlobalTransform>().m_value[3];
		auto& internodeData = i.GetPrivateComponent<InternodeData>();
		for (auto& leafTransform : internodeData->m_leavesTransforms)
		{
			auto* leaf = doc.allocate_node(rapidxml::node_element, "Leaf");
			leaf->append_attribute(doc.allocate_attribute("id", doc.allocate_string(std::to_string(counter).c_str())));
			counter++;
			leaves->append_node(leaf);

			auto* nodeAtt = doc.allocate_node(rapidxml::node_element, "Node");
			nodeAtt->append_attribute(doc.allocate_attribute("id", doc.allocate_string(std::to_string(i.m_index).c_str())));
			leaf->append_node(nodeAtt);

			auto* posAtt = doc.allocate_node(rapidxml::node_element, "Center");
			posAtt->append_attribute(doc.allocate_attribute("x", doc.allocate_string(std::to_string(nodePos.x).c_str())));
			posAtt->append_attribute(doc.allocate_attribute("y", doc.allocate_string(std::to_string(nodePos.y).c_str())));
			posAtt->append_attribute(doc.allocate_attribute("z", doc.allocate_string(std::to_string(nodePos.z).c_str())));
			leaf->append_node(posAtt);

			Transform transform;
			transform.m_value = leafTransform;
			auto rotation = transform.GetRotation();

			auto* frontAtt = doc.allocate_node(rapidxml::node_element, "Forward");
			frontAtt->append_attribute(doc.allocate_attribute("x", doc.allocate_string(std::to_string((rotation * glm::vec3(0, 0, -1)).x).c_str())));
			frontAtt->append_attribute(doc.allocate_attribute("y", doc.allocate_string(std::to_string((rotation * glm::vec3(0, 0, -1)).y).c_str())));
			frontAtt->append_attribute(doc.allocate_attribute("z", doc.allocate_string(std::to_string((rotation * glm::vec3(0, 0, -1)).z).c_str())));
			leaf->append_node(frontAtt);

			auto* leftAtt = doc.allocate_node(rapidxml::node_element, "Left");
			leftAtt->append_attribute(doc.allocate_attribute("x", doc.allocate_string(std::to_string((rotation * glm::vec3(1, 0, 0)).x).c_str())));
			leftAtt->append_attribute(doc.allocate_attribute("y", doc.allocate_string(std::to_string((rotation * glm::vec3(1, 0, 0)).y).c_str())));
			leftAtt->append_attribute(doc.allocate_attribute("z", doc.allocate_string(std::to_string((rotation * glm::vec3(1, 0, 0)).z).c_str())));
			leaf->append_node(leftAtt);

			auto* centerAtt = doc.allocate_node(rapidxml::node_element, "Position");
			centerAtt->append_attribute(doc.allocate_attribute("x", doc.allocate_string(std::to_string(leafTransform[3].x).c_str())));
			centerAtt->append_attribute(doc.allocate_attribute("y", doc.allocate_string(std::to_string(leafTransform[3].y).c_str())));
			centerAtt->append_attribute(doc.allocate_attribute("z", doc.allocate_string(std::to_string(leafTransform[3].z).c_str())));
			leaf->append_node(centerAtt);

			auto* sizeAtt = doc.allocate_node(rapidxml::node_element, "Size");
			sizeAtt->append_attribute(doc.allocate_attribute("x", doc.allocate_string(std::to_string(0.105).c_str())));
			sizeAtt->append_attribute(doc.allocate_attribute("y", doc.allocate_string(std::to_string(0.1155).c_str())));
			leaf->append_node(sizeAtt);

			auto* distAtt = doc.allocate_node(rapidxml::node_element, "Dist");
			distAtt->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(glm::distance(nodePos, glm::vec3(leafTransform[3]))).c_str())));
			leaf->append_node(distAtt);
		}
	}
}
