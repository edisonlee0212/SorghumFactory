#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API LeafSegment {
public:
  glm::vec3 m_position;
  glm::vec3 m_front;
  glm::vec3 m_up;
  glm::quat m_rotation;
  float m_surfacePush = 0.0f;
  float m_leafHalfWidth;
  float m_theta;
  float m_radius;
  float m_leftFlatness = 0.0f;
  float m_leftFlatnessFactor = 1.0f;
  float m_rightFlatness = 0.0f;
  float m_rightFlatnessFactor = 1.0f;
  bool m_isLeaf;
  LeafSegment(glm::vec3 position, glm::vec3 up, glm::vec3 front,
              float leafHalfWidth, float theta, bool isLeaf, float surfacePush,
              float leftFlatness = 0.0f, float rightFlatness = 0.0f,
              float leftFlatnessFactor = 1.0f,
              float rightFlatnessFactor = 1.0f);

  glm::vec3 GetPoint(float angle);
};
} // namespace PlantFactory