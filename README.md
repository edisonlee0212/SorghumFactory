# SorghumFactory
SorghumFactory is an application of UniEngine that provides a procedural sorghum model that will be used for illumination estimation of sorghum field.

## Getting Started
The project is a CMake project. For project editing, code inspections, Visual Studio 2017 or 2019 is recommanded. Simply clone/download the project files and open the folder as project in Visual Studio and you are ready.
To directly build the project, scripts under the root folder build.cmd (for Windows) and build.sh (for Linux) are provided for building with single command line.
E.g. For Linux, the command may be :
 - bash build.sh (build in default settings)
 - bash build.sh --clean release (clean and build in release mode)
Please visit script for further details.
## Main features
 - Procedural Sorghum model:
    - SorghumStateGenerator(SPD) that allows targeted user to create procedural sorghums without any prior knowledge about programming. 
    - SorghumField for instantiating a field of sorghums based on different pattern with SPD.
    - Plant scale illumination estimation with ray tracer facility.
    - Video demo: 
    - [![Procedural Sorghum](https://img.youtube.com/vi/AnWrYYsf0Ns/0.jpg)](https://www.youtube.com/watch?v=AnWrYYsf0Ns)
 - Automated data generation pipeline for procedural sorghums. Users are allowed to generate hundreds of sorghums and store data including RGB image, Depth image, OBJ model, semantic mask, etc.
