#pragma once
#include <ICurve.hpp>
#include <LeafSegment.hpp>
#include <sorghum_factory_export.h>
#include "ProceduralSorghumDescriptor.hpp"
using namespace UniEngine;
namespace SorghumFactory {
struct SORGHUM_FACTORY_API SplineNode {
  glm::vec3 m_position;
  float m_theta;
  float m_width;
  glm::vec3 m_axis;
  bool m_isLeaf;
  float m_surfacePush = 0.0f;
  SplineNode(glm::vec3 position, float angle, float width, glm::vec3 axis,
            bool isLeaf, float surfacePush);
  SplineNode();
};

class SORGHUM_FACTORY_API Spline : public IPrivateComponent {
  void GenerateLeafGeometry(const ProceduralStemState & stemState, const ProceduralLeafState & leafState, int nodeAmount);
  void GenerateStemGeometry(const ProceduralStemState & stemState, int nodeAmount);
public:
  //The "normal" direction of the leaf.
  glm::vec3 m_left;
  //Spline representation from Mathieu's skeleton
  std::vector<BezierCurve> m_curves;

  //Geometry generation
  std::vector<SplineNode> m_nodes;
  std::vector<LeafSegment> m_segments;
  std::vector<Vertex> m_vertices;
  std::vector<glm::uvec3> m_triangles;
  glm::vec4 m_vertexColor = glm::vec4(0, 1, 0, 1);
  //Import from Mathieu's procedural skeleton
  void Import(std::ifstream &stream);
  [[nodiscard]] glm::vec3 EvaluatePointFromCurves(float point) const;
  [[nodiscard]] glm::vec3 EvaluateAxisFromCurves(float point) const;

  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void Copy(const std::shared_ptr<Spline> &target);
  void FormStem(const ProceduralStemState & stemState, int nodeAmount);
  void FormLeaf(const ProceduralStemState & stemState, const ProceduralLeafState & leafState, int nodeAmount);

};

} // namespace SorghumFactory