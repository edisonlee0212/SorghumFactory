#include <SorghumParameters.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>
#include <rapidxml_utils.hpp>

void PlantFactory::SorghumParameters::OnGui() {
  ImGui::DragInt("Leaf count", &m_leafCount, 0.01f);
  ImGui::DragFloat2("Branch angle mean/var", &m_branchAngleMean, 0.01f);
  ImGui::DragFloat2("Roll angle var/dist", &m_rollAngleVariance, 0.01f);
  ImGui::DragFloat2("Internode length mean/var", &m_internodeLength, 0.01f);
  ImGui::DragFloat2("Gravity bending/increase", &m_leafGravityBending, 0.01f);
  ImGui::DragFloat2("Leaf length/Decrease", &m_leafLengthBase, 0.01f);
}

void PlantFactory::SorghumParameters::Serialize(const std::string &path) const {
  std::ofstream ofs;
  ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
  if (!ofs.is_open()) {
    Debug::Error("Can't open file!");
    return;
  }
  rapidxml::xml_document<> doc;
  auto *type = doc.allocate_node(rapidxml::node_doctype, 0, "Parameters");
  doc.append_node(type);
  auto *param = doc.allocate_node(rapidxml::node_element, "Parameters",
                                  "LateralBudPerNode");
  doc.append_node(param);

  auto *leafCount = doc.allocate_node(rapidxml::node_element, "LeafCount", "");
  param->append_node(leafCount);
  leafCount->append_attribute(doc.allocate_attribute(
      "value", doc.allocate_string(std::to_string(m_leafCount).c_str())));

  auto *branchAngleMean =
      doc.allocate_node(rapidxml::node_element, "BranchAngle", "");
  param->append_node(branchAngleMean);
  branchAngleMean->append_attribute(doc.allocate_attribute(
      "value", doc.allocate_string(std::to_string(m_branchAngleMean).c_str())));

  auto *branchAngleVariance =
      doc.allocate_node(rapidxml::node_element, "BranchAngle", "");
  param->append_node(branchAngleVariance);
  branchAngleVariance->append_attribute(doc.allocate_attribute(
      "value",
      doc.allocate_string(std::to_string(m_branchAngleVariance).c_str())));

  auto *rollAngleVariance =
      doc.allocate_node(rapidxml::node_element, "RollAngleVariance", "");
  param->append_node(rollAngleVariance);
  rollAngleVariance->append_attribute(doc.allocate_attribute(
      "value",
      doc.allocate_string(std::to_string(m_rollAngleVariance).c_str())));

  auto *rollAngleVarianceDistanceFactor = doc.allocate_node(
      rapidxml::node_element, "RollAngleVarianceDistanceFactor", "");
  param->append_node(rollAngleVarianceDistanceFactor);
  rollAngleVarianceDistanceFactor->append_attribute(doc.allocate_attribute(
      "value", doc.allocate_string(
                   std::to_string(m_rollAngleVarianceDistanceFactor).c_str())));

  auto *internodeLength =
      doc.allocate_node(rapidxml::node_element, "InternodeLength", "");
  param->append_node(internodeLength);
  internodeLength->append_attribute(doc.allocate_attribute(
      "value", doc.allocate_string(std::to_string(m_internodeLength).c_str())));

  auto *internodeLengthVariance =
      doc.allocate_node(rapidxml::node_element, "InternodeLengthVariance", "");
  param->append_node(internodeLengthVariance);
  internodeLengthVariance->append_attribute(doc.allocate_attribute(
      "value",
      doc.allocate_string(std::to_string(m_internodeLengthVariance).c_str())));

  auto *leafGravityBending =
      doc.allocate_node(rapidxml::node_element, "LeafGravityBending", "");
  param->append_node(leafGravityBending);
  leafGravityBending->append_attribute(doc.allocate_attribute(
      "value",
      doc.allocate_string(std::to_string(m_leafGravityBending).c_str())));

  auto *leafGravityBendingIncreaseFactor = doc.allocate_node(
      rapidxml::node_element, "LeafGravityBendingIncreaseFactor", "");
  param->append_node(leafGravityBendingIncreaseFactor);
  leafGravityBendingIncreaseFactor->append_attribute(doc.allocate_attribute(
      "value",
      doc.allocate_string(
          std::to_string(m_leafGravityBendingIncreaseFactor).c_str())));

  ofs << doc;
  ofs.flush();
  ofs.close();
}

void PlantFactory::SorghumParameters::Deserialize(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
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
    memcpy_s(c_content.data(), c_content.size() - 1, content.data(),
             content.size());
    c_content[content.size()] = 0;
    rapidxml::xml_document<> doc;
    doc.parse<0>(c_content.data());
    auto *param = doc.first_node("Parameters");
    m_leafCount =
        std::atoi(param->first_node("LeafCount")->first_attribute()->value());

    m_branchAngleMean = std::atof(
        param->first_node("BranchAngleMean")->first_attribute()->value());
    m_branchAngleVariance = std::atof(
        param->first_node("BranchAngleVariance")->first_attribute()->value());

    m_rollAngleVariance = std::atof(
        param->first_node("RollAngleVariance")->first_attribute()->value());
    m_rollAngleVarianceDistanceFactor =
        std::atof(param->first_node("RollAngleVarianceDistanceFactor")
                      ->first_attribute()
                      ->value());

    m_internodeLength = std::atof(
        param->first_node("InternodeLength")->first_attribute()->value());
    m_internodeLengthVariance =
        std::atof(param->first_node("InternodeLengthVariance")
                      ->first_attribute()
                      ->value());

    m_leafGravityBending = std::atof(
        param->first_node("LeafGravityBending")->first_attribute()->value());
    m_leafGravityBendingIncreaseFactor =
        std::atof(param->first_node("LeafGravityBendingIncreaseFactor")
                      ->first_attribute()
                      ->value());
  } catch (std::ifstream::failure e) {
    Debug::Error("Failed to open file");
  }
}
void PlantFactory::SorghumParameters::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_leafCount" << YAML::Value << m_leafCount;

  out << YAML::Key << "m_branchAngleMean" << YAML::Value << m_branchAngleMean;
  out << YAML::Key << "m_branchAngleVariance" << YAML::Value << m_branchAngleVariance;

  out << YAML::Key << "m_rollAngleVariance" << YAML::Value << m_rollAngleVariance;
  out << YAML::Key << "m_rollAngleVarianceDistanceFactor" << YAML::Value << m_rollAngleVarianceDistanceFactor;

  out << YAML::Key << "m_internodeLength" << YAML::Value << m_internodeLength;
  out << YAML::Key << "m_internodeLengthVariance" << YAML::Value << m_internodeLengthVariance;

  out << YAML::Key << "m_leafGravityBending" << YAML::Value << m_leafGravityBending;
  out << YAML::Key << "m_leafGravityBendingLevelFactor" << YAML::Value << m_leafGravityBendingLevelFactor;
  out << YAML::Key << "m_leafGravityBendingIncreaseFactor" << YAML::Value << m_leafGravityBendingIncreaseFactor;

  out << YAML::Key << "m_leafLengthBase" << YAML::Value << m_leafLengthBase;
}
void PlantFactory::SorghumParameters::Deserialize(const YAML::Node &in) {
  m_leafCount = in["m_leafCount"].as<int>();

  m_branchAngleMean = in["m_branchAngleMean"].as<float>();
  m_branchAngleVariance = in["m_branchAngleVariance"].as<float>();

  m_rollAngleVariance = in["m_rollAngleVariance"].as<float>();
  m_rollAngleVarianceDistanceFactor = in["m_rollAngleVarianceDistanceFactor"].as<float>();

  m_internodeLength = in["m_internodeLength"].as<float>();
  m_internodeLengthVariance = in["m_internodeLengthVariance"].as<float>();

  m_leafGravityBending = in["m_leafGravityBending"].as<float>();
  m_leafGravityBendingLevelFactor = in["m_leafGravityBendingLevelFactor"].as<float>();
  m_leafGravityBendingIncreaseFactor = in["m_leafGravityBendingIncreaseFactor"].as<float>();
  m_leafLengthBase = in["m_leafLengthBase"].as<float>();
}
