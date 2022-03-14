#pragma once
#include "ProceduralSorghum.hpp"
#include <SorghumStateGenerator.hpp>
#include <sorghum_factory_export.h>
#include <ICurve.hpp>
#include <LeafSegment.hpp>
#include <sorghum_factory_export.h>
#include "Spline.hpp"
using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API StemData : public IPrivateComponent {
  void GenerateStemGeometry();
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
  void Copy(const std::shared_ptr<StemData> &target);
  void FormStem(const SorghumStatePair & sorghumStatePair);
  void OnInspect() override;
  void OnDestroy() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
} // namespace SorghumFactory