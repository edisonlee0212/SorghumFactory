#pragma once
#include <ICurve.hpp>
#include <LeafSegment.hpp>
#include <sorghum_factory_export.h>
#include "ProceduralSorghum.hpp"
using namespace UniEngine;
namespace SorghumFactory {
struct SORGHUM_FACTORY_API SplineNode {
  glm::vec3 m_position;
  float m_theta;
  float m_width;
  float m_waviness;
  glm::vec3 m_axis;
  bool m_isLeaf;
  float m_range;
  float m_surfacePush = 0.0f;

  SplineNode(glm::vec3 position, float angle, float width, float waviness, glm::vec3 axis,
            bool isLeaf, float surfacePush, float range);
  SplineNode();
};
} // namespace SorghumFactory