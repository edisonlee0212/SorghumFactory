#include <TreeParameters.hpp>
#include <rapidxml_print.hpp>
#include <rapidxml_utils.hpp>
#include <rapidxml.hpp>

using namespace PlantFactory;

#pragma region TreeParameters
void TreeParameters::OnGui()
{
	ImGui::DragInt("LateralBudPerNode", &m_lateralBudPerNode, 1, 1, 3);
	ImGui::DragFloat2("Apical Angle M/Var", &m_apicalAngleMean, 0.01f);
	ImGui::DragFloat2("Branching Angle M/Var", &m_branchingAngleMean, 0.01f);
	ImGui::DragFloat2("Roll Angle M/Var", &m_rollAngleMean, 0.01f);
	ImGui::DragFloat("Internode Len Base", &m_internodeLengthBase, 0.01f);

	ImGui::DragFloat2("Light requirement A/L", &m_apicalIlluminationRequirement, 0.01f);
	ImGui::Text("Favors to older buds");
	ImGui::DragFloat2("Inhibitor Base/Dis", &m_inhibitorBase, 0.01f);
	ImGui::Text("Favors to main child");
	ImGui::DragFloat("Apical Control Lvl", &m_apicalControlLevelFactor, 0.01f);
	ImGui::DragFloat2("Resource weight Apical/Var", &m_resourceWeightApical, 0.01f);
	ImGui::DragFloat3("Height Decrease Min/Base/Fac", &m_heightResourceHeightDecreaseMin, 0.00001f, 0.0f, 4.0f, "%.5f");
	const float maxHeight = glm::pow((1.0f - m_heightResourceHeightDecreaseMin) / m_heightResourceHeightDecreaseBase, 1.0f / m_heightResourceHeightDecreaseFactor);
	const float max95Height = glm::pow(0.95f / m_heightResourceHeightDecreaseBase, 1.0f / m_heightResourceHeightDecreaseFactor);
	ImGui::Text("95 height: %.2f", max95Height);
	ImGui::Text("Maximum tree height: %.2f", maxHeight);
	
	ImGui::DragFloat("Avoidance Angle", &m_avoidanceAngle, 0.01f);
	ImGui::DragFloat("Phototropism", &m_phototropism, 0.01f);
	ImGui::DragFloat("Gravitropism", &m_gravitropism, 0.01f);

	ImGui::DragFloat3("RandomCutOff /Age/Max", &m_randomCutOff, 0.0001f, -1.0f, 1.0f, "%.5f");
	const float maxAgeBeforeMaxCutOff = (m_randomCutOffMax - m_randomCutOff) / m_randomCutOffAgeFactor;
	ImGui::Text("Max age before reaching max: %.2f", maxAgeBeforeMaxCutOff);
	ImGui::DragFloat("LowBranchCutOff", &m_lowBranchCutOff, 0.01f);

	ImGui::DragFloat2("Thickness End/Fac", &m_endNodeThickness, 0.01f);

	ImGui::DragFloat3("Bending Fac/Thick/Max", &m_gravityBendingFactor, 0.01f);
	
	ImGui::DragInt("Tree type", &m_treeType);
}

void TreeParameters::Serialize(const std::string& path) const
{
	std::ofstream ofs;
	ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
	if (!ofs.is_open())
	{
		Debug::Error("Can't open file!");
		return;
	}
	rapidxml::xml_document<> doc;
	auto* type = doc.allocate_node(rapidxml::node_doctype, 0, "Parameters");
	doc.append_node(type);
	auto* param = doc.allocate_node(rapidxml::node_element, "Parameters", "LateralBudPerNode");
	doc.append_node(param);

	auto* lateralBudPerNode = doc.allocate_node(rapidxml::node_element, "LateralBudPerNode", "");
	param->append_node(lateralBudPerNode);
	lateralBudPerNode->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_lateralBudPerNode).c_str())));

	auto* apicalAngleMean = doc.allocate_node(rapidxml::node_element, "ApicalAngleMean", "");
	param->append_node(apicalAngleMean);
	apicalAngleMean->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_apicalAngleMean).c_str())));

	auto* apicalAngleVariance = doc.allocate_node(rapidxml::node_element, "ApicalAngleVariance", "");
	param->append_node(apicalAngleVariance);
	apicalAngleVariance->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_apicalAngleVariance).c_str())));

	auto* branchingAngleMean = doc.allocate_node(rapidxml::node_element, "BranchingAngleMean", "");
	param->append_node(branchingAngleMean);
	branchingAngleMean->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_branchingAngleMean).c_str())));

	auto* branchingAngleVariance = doc.allocate_node(rapidxml::node_element, "BranchingAngleVariance", "");
	param->append_node(branchingAngleVariance);
	branchingAngleVariance->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_branchingAngleVariance).c_str())));

	auto* rollAngleMean = doc.allocate_node(rapidxml::node_element, "RollAngleMean", "");
	param->append_node(rollAngleMean);
	rollAngleMean->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_rollAngleMean).c_str())));

	auto* rollAngleVariance = doc.allocate_node(rapidxml::node_element, "RollAngleVariance", "");
	param->append_node(rollAngleVariance);
	rollAngleVariance->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_rollAngleVariance).c_str())));

	auto* internodeLengthBase = doc.allocate_node(rapidxml::node_element, "InternodeLengthBase", "");
	param->append_node(internodeLengthBase);
	internodeLengthBase->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_internodeLengthBase).c_str())));

	auto* apicalIlluminationRequirement = doc.allocate_node(rapidxml::node_element, "ApicalIlluminationRequirement", "");
	param->append_node(apicalIlluminationRequirement);
	apicalIlluminationRequirement->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_apicalIlluminationRequirement).c_str())));

	auto* lateralIlluminationRequirement = doc.allocate_node(rapidxml::node_element, "LateralIlluminationRequirement", "");
	param->append_node(lateralIlluminationRequirement);
	lateralIlluminationRequirement->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_lateralIlluminationRequirement).c_str())));

	auto* inhibitorBase = doc.allocate_node(rapidxml::node_element, "InhibitorBase", "");
	param->append_node(inhibitorBase);
	inhibitorBase->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_inhibitorBase).c_str())));

	auto* inhibitorDistanceFactor = doc.allocate_node(rapidxml::node_element, "InhibitorDistanceFactor", "");
	param->append_node(inhibitorDistanceFactor);
	inhibitorDistanceFactor->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_inhibitorDistanceFactor).c_str())));

	auto* resourceWeightApical = doc.allocate_node(rapidxml::node_element, "ResourceWeightApical", "");
	param->append_node(resourceWeightApical);
	resourceWeightApical->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_resourceWeightApical).c_str())));

	auto* resourceWeightVariance = doc.allocate_node(rapidxml::node_element, "ResourceWeightVariance", "");
	param->append_node(resourceWeightVariance);
	resourceWeightVariance->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_resourceWeightVariance).c_str())));

	auto* apicalControlLevelFactor = doc.allocate_node(rapidxml::node_element, "ApicalControlLevelFactor", "");
	param->append_node(apicalControlLevelFactor);
	apicalControlLevelFactor->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_apicalControlLevelFactor).c_str())));

	auto* avoidanceAngle = doc.allocate_node(rapidxml::node_element, "AvoidanceAngle", "");
	param->append_node(avoidanceAngle);
	avoidanceAngle->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_avoidanceAngle).c_str())));

	auto* heightResourceHeightDecreaseMin = doc.allocate_node(rapidxml::node_element, "HeightResourceHeightDecreaseMin", "");
	param->append_node(heightResourceHeightDecreaseMin);
	heightResourceHeightDecreaseMin->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_heightResourceHeightDecreaseMin).c_str())));
	
	auto* heightResourceHeightDecreaseBase = doc.allocate_node(rapidxml::node_element, "HeightResourceHeightDecreaseBase", "");
	param->append_node(heightResourceHeightDecreaseBase);
	heightResourceHeightDecreaseBase->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_heightResourceHeightDecreaseBase).c_str())));

	auto* heightResourceHeightDecreaseFactor = doc.allocate_node(rapidxml::node_element, "HeightResourceHeightDecreaseFactor", "");
	param->append_node(heightResourceHeightDecreaseFactor);
	heightResourceHeightDecreaseFactor->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_heightResourceHeightDecreaseFactor).c_str())));
	
	auto* phototropism = doc.allocate_node(rapidxml::node_element, "Phototropism", "");
	param->append_node(phototropism);
	phototropism->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_phototropism).c_str())));

	auto* gravitropism = doc.allocate_node(rapidxml::node_element, "Gravitropism", "");
	param->append_node(gravitropism);
	gravitropism->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_gravitropism).c_str())));

	auto* randomCutOff = doc.allocate_node(rapidxml::node_element, "RandomCutOff", "");
	param->append_node(randomCutOff);
	randomCutOff->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_randomCutOff).c_str())));

	auto* randomCutOffAgeFactor = doc.allocate_node(rapidxml::node_element, "RandomCutOffAgeFactor", "");
	param->append_node(randomCutOffAgeFactor);
	randomCutOffAgeFactor->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_randomCutOffAgeFactor).c_str())));

	auto* randomCutOffMax = doc.allocate_node(rapidxml::node_element, "RandomCutOffMax", "");
	param->append_node(randomCutOffMax);
	randomCutOffMax->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_randomCutOffMax).c_str())));

	auto* lowBranchCutOff = doc.allocate_node(rapidxml::node_element, "LowBranchCutOff", "");
	param->append_node(lowBranchCutOff);
	lowBranchCutOff->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_lowBranchCutOff).c_str())));
	
	auto* endNodeThickness = doc.allocate_node(rapidxml::node_element, "EndNodeThickness", "");
	param->append_node(endNodeThickness);
	endNodeThickness->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_endNodeThickness).c_str())));

	auto* thicknessControlFactor = doc.allocate_node(rapidxml::node_element, "ThicknessControlFactor", "");
	param->append_node(thicknessControlFactor);
	thicknessControlFactor->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_thicknessControlFactor).c_str())));

	auto* gravityBendingFactor = doc.allocate_node(rapidxml::node_element, "GravityBendingFactor", "");
	param->append_node(gravityBendingFactor);
	gravityBendingFactor->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_gravityBendingFactor).c_str())));

	auto* gravityBendingThicknessFactor = doc.allocate_node(rapidxml::node_element, "GravityBendingThicknessFactor", "");
	param->append_node(gravityBendingThicknessFactor);
	gravityBendingThicknessFactor->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_gravityBendingThicknessFactor).c_str())));

	auto* gravityBendingMax = doc.allocate_node(rapidxml::node_element, "GravityBendingMax", "");
	param->append_node(gravityBendingMax);
	gravityBendingMax->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_gravityBendingMax).c_str())));


	auto* treeType = doc.allocate_node(rapidxml::node_element, "TreeType", "");
	param->append_node(treeType);
	treeType->append_attribute(doc.allocate_attribute("value", doc.allocate_string(std::to_string(m_treeType).c_str())));

	ofs << doc;
	ofs.flush();
	ofs.close();
}

void TreeParameters::Deserialize(const std::string& path)
{
	std::ifstream file;
	file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		// open files
		file.open(path);
		std::stringstream stream;
		// read file's buffer contents into streams
		stream << file.rdbuf();
		// close file handlers
		file.close();
		// convert stream into string
		auto content = stream.str();
		std::vector<char> c_content;
		c_content.resize(content.size() + 1);
		memcpy_s(c_content.data(), c_content.size() - 1, content.data(), content.size());
		c_content[content.size()] = 0;
		rapidxml::xml_document<> doc;
		doc.parse<0>(c_content.data());
		auto* param = doc.first_node("Parameters");
		m_lateralBudPerNode = std::atoi(param->first_node("LateralBudPerNode")->first_attribute()->value());
		m_apicalAngleMean = std::atof(param->first_node("ApicalAngleMean")->first_attribute()->value());
		m_apicalAngleVariance = std::atof(param->first_node("ApicalAngleVariance")->first_attribute()->value());
		m_branchingAngleMean = std::atof(param->first_node("BranchingAngleMean")->first_attribute()->value());
		m_branchingAngleVariance = std::atof(param->first_node("BranchingAngleVariance")->first_attribute()->value());
		m_rollAngleMean = std::atof(param->first_node("RollAngleMean")->first_attribute()->value());
		m_rollAngleVariance = std::atof(param->first_node("RollAngleVariance")->first_attribute()->value());
		m_internodeLengthBase = std::atof(param->first_node("InternodeLengthBase")->first_attribute()->value());
		m_apicalIlluminationRequirement = std::atof(param->first_node("ApicalIlluminationRequirement")->first_attribute()->value());
		m_lateralIlluminationRequirement = std::atof(param->first_node("LateralIlluminationRequirement")->first_attribute()->value());
		m_inhibitorBase = std::atof(param->first_node("InhibitorBase")->first_attribute()->value());
		m_inhibitorDistanceFactor = std::atof(param->first_node("InhibitorDistanceFactor")->first_attribute()->value());
		m_resourceWeightApical = std::atof(param->first_node("ResourceWeightApical")->first_attribute()->value());
		m_resourceWeightVariance = std::atof(param->first_node("ResourceWeightVariance")->first_attribute()->value());

		m_apicalControlLevelFactor = std::atof(param->first_node("ApicalControlLevelFactor")->first_attribute()->value());
		
		m_heightResourceHeightDecreaseMin = std::atof(param->first_node("HeightResourceHeightDecreaseMin")->first_attribute()->value());
		m_heightResourceHeightDecreaseBase = std::atof(param->first_node("HeightResourceHeightDecreaseBase")->first_attribute()->value());
		m_heightResourceHeightDecreaseFactor = std::atof(param->first_node("HeightResourceHeightDecreaseFactor")->first_attribute()->value());
		
		m_avoidanceAngle = std::atof(param->first_node("AvoidanceAngle")->first_attribute()->value());
		m_phototropism = std::atof(param->first_node("Phototropism")->first_attribute()->value());
		m_gravitropism = std::atof(param->first_node("Gravitropism")->first_attribute()->value());
		m_randomCutOff = std::atof(param->first_node("RandomCutOff")->first_attribute()->value());
		m_randomCutOffAgeFactor = std::atof(param->first_node("RandomCutOffAgeFactor")->first_attribute()->value());
		m_randomCutOffMax = std::atof(param->first_node("RandomCutOffMax")->first_attribute()->value());
		
		m_lowBranchCutOff = std::atof(param->first_node("LowBranchCutOff")->first_attribute()->value());
		m_endNodeThickness = std::atof(param->first_node("EndNodeThickness")->first_attribute()->value());
		m_thicknessControlFactor = std::atof(param->first_node("ThicknessControlFactor")->first_attribute()->value());
		m_gravityBendingFactor = std::atof(param->first_node("GravityBendingFactor")->first_attribute()->value());
		m_gravityBendingThicknessFactor = std::atof(param->first_node("GravityBendingThicknessFactor")->first_attribute()->value());
		m_gravityBendingMax = std::atof(param->first_node("GravityBendingMax")->first_attribute()->value());
		m_treeType = std::atoi(param->first_node("TreeType")->first_attribute()->value());
	}
	catch (std::ifstream::failure e)
	{
		Debug::Error("Failed to open file");
	}
}

#pragma endregion