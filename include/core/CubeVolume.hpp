#pragma once
#include <Volume.hpp>
using namespace UniEngine;
namespace PlantFactory
{
    class CubeVolume :
        public Volume
    {
    public:
        void ApplyMeshRendererBounds();
        CubeVolume();
        bool m_displayPoints = true;
        bool m_displayBounds = true;
        Bound m_minMaxBound;
    	void OnGui() override;
    	bool InVolume(const glm::vec3& position) override;
    	glm::vec3 GetRandomPoint() override;
    };
}