/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sstream>
#include <vulkan/vulkan.hpp>

extern std::vector<std::string> defaultSearchPaths;

#define STB_IMAGE_IMPLEMENTATION
#include "fileformats/stb_image.h"
#include "obj_loader.h"

#include "hello_vulkan.h"
#include "nvh//cameramanipulator.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"

#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"

#include "nvh/alignment.hpp"
#include "nvvk/shaders_vk.hpp"
#include <random>

// Holding the camera matrices
struct CameraMatrices
{
  nvmath::mat4f view;
  nvmath::mat4f proj;
  nvmath::mat4f viewInverse;
  // #VKRay
  nvmath::mat4f projInverse;
};

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const vk::Instance&       instance,
                        const vk::Device&         device,
                        const vk::PhysicalDevice& physicalDevice,
                        uint32_t                  queueFamily)
{
  AppBase::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(device, physicalDevice);
  m_debug.setup(m_device);
  m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const vk::CommandBuffer& cmdBuf)
{
  // Prepare new UBO contents on host.
  const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
  CameraMatrices hostUBO     = {};
  hostUBO.view               = CameraManip.getMatrix();
  hostUBO.proj = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // hostUBO.proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).
  hostUBO.viewInverse = nvmath::invert(hostUBO.view);
  // #VKRay
  hostUBO.projInverse = nvmath::invert(hostUBO.proj);

  // UBO on the device, and what stages access it.
  vk::Buffer deviceUBO = m_cameraMat.buffer;
  auto       uboUsageStages =
      vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eRayTracingShaderKHR;

  // Ensure that the modified UBO is not visible to previous frames.
  vk::BufferMemoryBarrier beforeBarrier;
  beforeBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderRead);
  beforeBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
  beforeBarrier.setBuffer(deviceUBO);
  beforeBarrier.setOffset(0);
  beforeBarrier.setSize(sizeof hostUBO);
  cmdBuf.pipelineBarrier(uboUsageStages, vk::PipelineStageFlagBits::eTransfer,
                         vk::DependencyFlagBits::eDeviceGroup, {}, {beforeBarrier}, {});

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  cmdBuf.updateBuffer<CameraMatrices>(m_cameraMat.buffer, 0, hostUBO);

  // Making sure the updated UBO will be visible.
  vk::BufferMemoryBarrier afterBarrier;
  afterBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
  afterBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  afterBarrier.setBuffer(deviceUBO);
  afterBarrier.setOffset(0);
  afterBarrier.setSize(sizeof hostUBO);
  cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, uboUsageStages,
                         vk::DependencyFlagBits::eDeviceGroup, {}, {afterBarrier}, {});
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  using vkDS     = vk::DescriptorSetLayoutBinding;
  using vkDT     = vk::DescriptorType;
  using vkSS     = vk::ShaderStageFlagBits;
  uint32_t nbTxt = static_cast<uint32_t>(m_textures.size());
  uint32_t nbObj = static_cast<uint32_t>(m_objModel.size());

  // Camera matrices (binding = 0)
  m_descSetLayoutBind.addBinding(
      vkDS(0, vkDT::eUniformBuffer, 1, vkSS::eVertex | vkSS::eRaygenKHR));
  // Materials (binding = 1)
  m_descSetLayoutBind.addBinding(vkDS(1, vkDT::eStorageBuffer, nbObj + 1,
                                      vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitKHR));
  // Scene description (binding = 2)
  m_descSetLayoutBind.addBinding(  //
      vkDS(2, vkDT::eStorageBuffer, 1, vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitKHR));
  // Textures (binding = 3)
  m_descSetLayoutBind.addBinding(
      vkDS(3, vkDT::eCombinedImageSampler, nbTxt, vkSS::eFragment | vkSS::eClosestHitKHR));
  // Materials Index (binding = 4)
  m_descSetLayoutBind.addBinding(
      vkDS(4, vkDT::eStorageBuffer, nbObj + 1, vkSS::eFragment | vkSS::eClosestHitKHR));
  // Storing vertices (binding = 5)
  m_descSetLayoutBind.addBinding(  //
      vkDS(5, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitKHR));
  // Storing indices (binding = 6)
  m_descSetLayoutBind.addBinding(  //
      vkDS(6, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitKHR));
  // Storing spheres (binding = 7)
  m_descSetLayoutBind.addBinding(  //
      vkDS(7, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eIntersectionKHR));
  // Storing beziers (binding = 8)
  m_descSetLayoutBind.addBinding(  //
      vkDS(8, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eIntersectionKHR));
  // Storing beziers (binding = 8)
  m_descSetLayoutBind.addBinding(  //
      vkDS(9, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eIntersectionKHR));


  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<vk::WriteDescriptorSet> writes;

  // Camera matrices and scene description
  vk::DescriptorBufferInfo dbiUnif{m_cameraMat.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 0, &dbiUnif));
  vk::DescriptorBufferInfo dbiSceneDesc{m_sceneDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 2, &dbiSceneDesc));

  // All material buffers, 1 buffer per OBJ
  std::vector<vk::DescriptorBufferInfo> dbiMat;
  std::vector<vk::DescriptorBufferInfo> dbiMatIdx;
  std::vector<vk::DescriptorBufferInfo> dbiVert;
  std::vector<vk::DescriptorBufferInfo> dbiIdx;
  for(auto& obj : m_objModel)
  {
    dbiMat.emplace_back(obj.matColorBuffer.buffer, 0, VK_WHOLE_SIZE);
    dbiMatIdx.emplace_back(obj.matIndexBuffer.buffer, 0, VK_WHOLE_SIZE);
    dbiVert.emplace_back(obj.vertexBuffer.buffer, 0, VK_WHOLE_SIZE);
    dbiIdx.emplace_back(obj.indexBuffer.buffer, 0, VK_WHOLE_SIZE);
  }
  dbiMat.emplace_back(m_spheresMatColorBuffer.buffer, 0, VK_WHOLE_SIZE);
  dbiMatIdx.emplace_back(m_spheresMatIndexBuffer.buffer, 0, VK_WHOLE_SIZE);

  dbiMat.emplace_back(m_beziersMatColorBuffer.buffer, 0, VK_WHOLE_SIZE);
  dbiMatIdx.emplace_back(m_bezierMatIndexBuffer.buffer, 0, VK_WHOLE_SIZE);

  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 1, dbiMat.data()));
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 4, dbiMatIdx.data()));
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 5, dbiVert.data()));
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 6, dbiIdx.data()));

  vk::DescriptorBufferInfo dbiSpheres{m_spheresBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 7, &dbiSpheres));

  vk::DescriptorBufferInfo dbiBeziers{m_beziersBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 8, &dbiBeziers));

  vk::DescriptorBufferInfo dbiHairs{m_hairsBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 9, &dbiHairs));


  // All texture samplers
  std::vector<vk::DescriptorImageInfo> diit;
  for(auto& texture : m_textures)
  {
    diit.emplace_back(texture.descriptor);
  }
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 3, diit.data()));

  // Writing the information
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  using vkSS = vk::ShaderStageFlagBits;

  vk::PushConstantRange pushConstantRanges = {vkSS::eVertex | vkSS::eFragment, 0,
                                              sizeof(ObjPushConstant)};

  // Creating the Pipeline Layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  vk::DescriptorSetLayout      descSetLayout(m_descSetLayout);
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&descSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(nvh::loadFile("spv/vert_shader.vert.spv", true, paths, true), vkSS::eVertex);
  gpb.addShader(nvh::loadFile("spv/frag_shader.frag.spv", true, paths, true), vkSS::eFragment);
  gpb.addBindingDescription({0, sizeof(VertexObj)});
  gpb.addAttributeDescriptions({{0, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, pos)},
                                {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, nrm)},
                                {2, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, color)},
                                {3, 0, vk::Format::eR32G32Sfloat, offsetof(VertexObj, texCoord)}});

  m_graphicsPipeline = gpb.createPipeline();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadModel(const std::string& filename, nvmath::mat4f transform)
{
  using vkBU = vk::BufferUsageFlagBits;

  LOGI("Loading File:  %s \n", filename.c_str());
  ObjLoader loader;
  loader.loadModel(filename);

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = nvmath::pow(m.ambient, 2.2f);
    m.diffuse  = nvmath::pow(m.diffuse, 2.2f);
    m.specular = nvmath::pow(m.specular, 2.2f);
  }

  ObjInstance instance;
  instance.objIndex    = static_cast<uint32_t>(m_objModel.size());
  instance.transform   = transform;
  instance.transformIT = nvmath::transpose(nvmath::invert(transform));
  instance.txtOffset   = static_cast<uint32_t>(m_textures.size());

  ObjModel model;
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();
  model.vertexBuffer =
      m_alloc.createBuffer(cmdBuf, loader.m_vertices,
                           vkBU::eVertexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                               | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);
  model.indexBuffer =
      m_alloc.createBuffer(cmdBuf, loader.m_indices,
                           vkBU::eIndexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                               | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, vkBU::eStorageBuffer);
  model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, vkBU::eStorageBuffer);
  // Creates all textures found
  createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  std::string objNb = std::to_string(instance.objIndex);
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb).c_str()));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb).c_str()));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb).c_str()));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb).c_str()));

  m_objModel.emplace_back(model);
  m_objInstance.emplace_back(instance);
}

void HelloVulkan::loadHairModel(const char* filename, cyHairFile& hairfile)
{

  // Load the hair model
  int result = hairfile.LoadFromFile(filename);
  // Check for errors
  switch(result)
  {
    case CY_HAIR_FILE_ERROR_CANT_OPEN_FILE:
      printf("Error: Cannot open hair file!\n");
      return;
    case CY_HAIR_FILE_ERROR_CANT_READ_HEADER:
      printf("Error: Cannot read hair file header!\n");
      return;
    case CY_HAIR_FILE_ERROR_WRONG_SIGNATURE:
      printf("Error: File has wrong signature!\n");
      return;
    case CY_HAIR_FILE_ERROR_READING_SEGMENTS:
      printf("Error: Cannot read hair segments!\n");
      return;
    case CY_HAIR_FILE_ERROR_READING_POINTS:
      printf("Error: Cannot read hair points!\n");
      return;
    case CY_HAIR_FILE_ERROR_READING_COLORS:
      printf("Error: Cannot read hair colors!\n");
      return;
    case CY_HAIR_FILE_ERROR_READING_THICKNESS:
      printf("Error: Cannot read hair thickness!\n");
      return;
    case CY_HAIR_FILE_ERROR_READING_TRANSPARENCY:
      printf("Error: Cannot read hair transparency!\n");
      return;
    default:
      printf("Hair file \"%s\" loaded.\n", filename);
  }
  int hairCount  = (hairfile.GetHeader().hair_count);
  int pointCount = (hairfile.GetHeader().point_count);
  printf("Number of hair strands = %d\n", hairCount);
  printf("Number of hair points = %d\n", pointCount);
  float* dirs = new float[3807654];
  // Compute directions
  if(hairfile.FillDirectionArray(dirs) == 0)
  {
    printf("Error: Cannot compute hair directions!\n");
  }
  nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();
#if 0
  using vkBU = vk::BufferUsageFlagBits;

  nvmath::mat4f transform = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

  ObjInstance instance;
  instance.objIndex    = static_cast<uint32_t>(m_objModel.size());
  instance.transform   = transform;
  instance.transformIT = nvmath::transpose(nvmath::invert(transform));
  instance.txtOffset   = static_cast<uint32_t>(m_textures.size());

  ObjLoader loader;
  ObjModel  model;
  model.nbIndices  = static_cast<uint32_t>(hairfile.GetHeader().point_count);
  model.nbVertices = static_cast<uint32_t>(hairfile.GetHeader().point_count);
  // Create the buffers on Device and copy vertices, indices and materials
  
  std::vector<VertexObj> myVertices;
	
  for(int i = 0; i < hairfile.GetHeader().point_count - 2; i += 3)
  {
    myVertices.push_back(VertexObj{
        nvmath::vec3(hairfile.GetPointsArray()[i], hairfile.GetPointsArray()[i + 1],
                     hairfile.GetPointsArray()[i + 2]),
        nvmath::vec3(0.0f, 1.0f, 1.0f), nvmath::vec3(0.0f, 0.0f, 1.0f), nvmath::vec2(1.0f, 0.0f)});
  }
  model.vertexBuffer =
      m_alloc.createBuffer(cmdBuf, myVertices,
                           vkBU::eVertexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                               | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);

  std::vector<uint32_t> myIndices;
  for(int i = 0; i < hairfile.GetHeader().point_count; i++)
  {
    myIndices.push_back(*reinterpret_cast<uint32_t const*>(hairfile.GetSegmentsArray()));
  }
  model.indexBuffer =
      m_alloc.createBuffer(cmdBuf, myIndices,
                           vkBU::eIndexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                               | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);

  std::vector<MaterialObj> myColors;
  for(int i = 0; i < hairfile.GetHeader().point_count - 2; i += 3)
  {
    myColors.push_back(
        MaterialObj{nvmath::vec3(hairfile.GetColorsArray()[i], hairfile.GetColorsArray()[i + 1],
                                 hairfile.GetColorsArray()[i + 2]),
                    nvmath::vec3(hairfile.GetColorsArray()[i], hairfile.GetColorsArray()[i + 1],
                                 hairfile.GetColorsArray()[i + 2]),
                    nvmath::vec3(1.0f, 1.0f, 1.0f), nvmath::vec3(0.0f, 0.0f, 0.0f),
                    nvmath::vec3(0.0f, 0.0f, 0.10), 0.f, 1.0f, 1.f, 0, -1});
  }

  /*std::vector<MaterialObj> m_materials = MaterialObj({nvmath::vec3f(0.1f, 0.1f, 0.1f),
                                          nvmath::vec3f(0.7f, 0.7f, 0.7f),
                                          nvmath::vec3f(1.0f, 1.0f, 1.0f),
                                          nvmath::vec3f(0.0f, 0.0f, 0.0f),
                                          nvmath::vec3f(0.0f, 0.0f, 0.10),
                                          0.f,
                                          1.0f,
                                          1.f,
                                          0,
                                          -1}); */
  std::vector<int32_t> m_matIndx = {0};
  model.matColorBuffer           = m_alloc.createBuffer(cmdBuf, myColors, vkBU::eStorageBuffer);
  model.matIndexBuffer           = m_alloc.createBuffer(cmdBuf, m_matIndx, vkBU::eStorageBuffer);

  // Creates all textures found
  /*createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();*/

  std::string objNb = std::to_string(instance.objIndex);
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb).c_str()));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb).c_str()));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb).c_str()));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb).c_str()));


  m_objModel.emplace_back(model);
  m_objInstance.emplace_back(instance);
#endif

#if 1
  std::vector<Aabb>     hairAabbs;
  int                   pointIndex = 0;
  float*                vertices   = hairfile.GetPointsArray();
  float*                colors     = hairfile.GetColorsArray();
  const unsigned short* segments   = hairfile.GetSegmentsArray();
  for(int i = 0; i < hairCount; i++)
  {
    for(int j = 0; j < segments[i]; j++)
    {
      m_hairs.push_back(Hair {
			nvmath::vec3(vertices[pointIndex + j * 3], vertices[pointIndex + 1 + j * 3],
                     vertices[pointIndex + 2 + j * 3])
				/ 3.0f,
            nvmath::vec3(vertices[pointIndex + 3 + j * 3], vertices[pointIndex + 4 + j * 3],
                         vertices[pointIndex + 5 + j * 3])
                / 3.0f,
            nvmath::vec3(colors[pointIndex + j * 3], colors[pointIndex + 1 + j * 3],
                         colors[pointIndex + 2 + j * 3]),
            nvmath::vec3(colors[pointIndex + 3 + j * 3], colors[pointIndex + 4 + j * 3],
                         colors[pointIndex + 5 + j * 3]),
            nvmath::vec3(dirs[pointIndex + j * 3], dirs[pointIndex + 1 + j * 3],
                            dirs[pointIndex + 2 + j * 3]),
            nvmath::vec3(colors[pointIndex + 3 + j * 3], colors[pointIndex + 4 + j * 3],
                            colors[pointIndex + 5 + j * 3]),
            0.1f
      });
    }
    pointIndex += (segments[i] + 1) * 3;
  }
#endif

#if 1
  //Box for hair

  for(const auto hair : m_hairs)
  {
    nvmath::vec3 a = hair.p1 - hair.p0;
    nvmath::vec3 extent =
        nvmath::vec3(hair.thickness / 2) * /*sqrt*/ (nvmath::vec3(1.0f) - nvmath::normalize(a));
    Aabb hairAabb{nvmath::nv_min(hair.p0 - extent, hair.p1 - extent),
                  nvmath::nv_max(hair.p0 + extent, hair.p1 + extent)};
    hairAabbs.emplace_back(hairAabb);
  }
#endif

#if 1
  m_hairsBuffer = m_alloc.createBuffer(cmdBuf, m_hairs, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_hairsAabbBuffer =
      m_alloc.createBuffer(cmdBuf, hairAabbs, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  genCmdBuf.submitAndWait(cmdBuf);

  // Debug information
  m_debug.setObjectName(m_hairsBuffer.buffer, "hairs");
  m_debug.setObjectName(m_hairsAabbBuffer.buffer, "hairsAabb");
#endif
}

//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  using vkMP = vk::MemoryPropertyFlagBits;

  m_cameraMat = m_alloc.createBuffer(sizeof(CameraMatrices),
                                     vkBU::eUniformBuffer | vkBU::eTransferDst, vkMP::eDeviceLocal);
  m_debug.setObjectName(m_cameraMat.buffer, "cameraMat");
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void HelloVulkan::createSceneDescriptionBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_sceneDesc = m_alloc.createBuffer(cmdBuf, m_objInstance, vkBU::eStorageBuffer);
  cmdGen.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  m_debug.setObjectName(m_sceneDesc.buffer, "sceneDesc");
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const vk::CommandBuffer&        cmdBuf,
                                      const std::vector<std::string>& textures)
{
  using vkIU = vk::ImageUsageFlagBits;

  vk::SamplerCreateInfo samplerCreateInfo{
      {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  samplerCreateInfo.setMaxLod(FLT_MAX);
  vk::Format format = vk::Format::eR8G8B8A8Srgb;

  // If no textures are present, create a dummy one to accommodate the pipeline layout
  if(textures.empty() && m_textures.empty())
  {
    nvvk::Texture texture;

    std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
    vk::DeviceSize         bufferSize      = sizeof(color);
    auto                   imgSize         = vk::Extent2D(1, 1);
    auto                   imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format);

    // Creating the dummy texture
    nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    texture                        = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

    // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    nvvk::cmdBarrierImageLayout(cmdBuf, texture.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eShaderReadOnlyOptimal);
    m_textures.push_back(texture);
  }
  else
  {
    // Uploading all images
    for(const auto& texture : textures)
    {
      std::stringstream o;
      int               texWidth, texHeight, texChannels;
      o << "media/textures/" << texture;
      std::string txtFile = nvh::findFile(o.str(), defaultSearchPaths, true);

      stbi_uc* stbi_pixels =
          stbi_load(txtFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

      std::array<stbi_uc, 4> color{255u, 0u, 255u, 255u};

      stbi_uc* pixels = stbi_pixels;
      // Handle failure
      if(!stbi_pixels)
      {
        texWidth = texHeight = 1;
        texChannels          = 4;
        pixels               = reinterpret_cast<stbi_uc*>(color.data());
      }

      vk::DeviceSize bufferSize = static_cast<uint64_t>(texWidth) * texHeight * sizeof(uint8_t) * 4;
      auto           imgSize    = vk::Extent2D(texWidth, texHeight);
      auto imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

      {
        nvvk::ImageDedicated image =
            m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);
        nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
        vk::ImageViewCreateInfo ivInfo =
            nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
        nvvk::Texture texture = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

        m_textures.push_back(texture);
      }

      stbi_image_free(stbi_pixels);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  m_device.destroy(m_graphicsPipeline);
  m_device.destroy(m_pipelineLayout);
  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);
  m_alloc.destroy(m_cameraMat);
  m_alloc.destroy(m_sceneDesc);

  for(auto& m : m_objModel)
  {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
    m_alloc.destroy(m.matIndexBuffer);
  }

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  //#Post
  m_device.destroy(m_postPipeline);
  m_device.destroy(m_postPipelineLayout);
  m_device.destroy(m_postDescPool);
  m_device.destroy(m_postDescSetLayout);
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  m_device.destroy(m_offscreenRenderPass);
  m_device.destroy(m_offscreenFramebuffer);

  // #VKRay
  m_rtBuilder.destroy();
  m_device.destroy(m_rtDescPool);
  m_device.destroy(m_rtDescSetLayout);
  m_device.destroy(m_rtPipeline);
  m_device.destroy(m_rtPipelineLayout);
  m_alloc.destroy(m_rtSBTBuffer);


  m_alloc.destroy(m_spheresBuffer);
  m_alloc.destroy(m_beziersBuffer);
  m_alloc.destroy(m_hairsBuffer);
  m_alloc.destroy(m_spheresAabbBuffer);
  m_alloc.destroy(m_bezierAabbBuffer);
  m_alloc.destroy(m_hairsAabbBuffer);
  m_alloc.destroy(m_spheresMatColorBuffer);
  m_alloc.destroy(m_beziersMatColorBuffer);
  m_alloc.destroy(m_spheresMatIndexBuffer);
  m_alloc.destroy(m_bezierMatIndexBuffer);

  m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const vk::CommandBuffer& cmdBuf)
{
  using vkPBP = vk::PipelineBindPoint;
  using vkSS  = vk::ShaderStageFlagBits;
  vk::DeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

  // Drawing all triangles
  cmdBuf.bindPipeline(vkPBP::eGraphics, m_graphicsPipeline);
  cmdBuf.bindDescriptorSets(vkPBP::eGraphics, m_pipelineLayout, 0, {m_descSet}, {});
  for(int i = 0; i < m_objInstance.size(); ++i)
  {
    auto& inst                = m_objInstance[i];
    auto& model               = m_objModel[inst.objIndex];
    m_pushConstant.instanceId = i;  // Telling which instance is drawn
    cmdBuf.pushConstants<ObjPushConstant>(m_pipelineLayout, vkSS::eVertex | vkSS::eFragment, 0,
                                          m_pushConstant);

    cmdBuf.bindVertexBuffers(0, {model.vertexBuffer.buffer}, {offset});
    cmdBuf.bindIndexBuffer(model.indexBuffer.buffer, 0, vk::IndexType::eUint32);
    cmdBuf.drawIndexed(model.nbIndices, 1, 0, 0, 0);
  }

  // cmdBuf.bindVertexBuffers(0, {});

  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  createOffscreenRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
}

//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                       vk::ImageUsageFlagBits::eColorAttachment
                                                           | vk::ImageUsageFlagBits::eSampled
                                                           | vk::ImageUsageFlagBits::eStorage);


    nvvk::Image             image  = m_alloc.createImage(colorCreateInfo);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    m_offscreenColor               = m_alloc.createTexture(image, ivInfo, vk::SamplerCreateInfo());
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo =
      nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat,
                                  vk::ImageUsageFlagBits::eDepthStencilAttachment);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);

    vk::ImageViewCreateInfo depthStencilView;
    depthStencilView.setViewType(vk::ImageViewType::e2D);
    depthStencilView.setFormat(m_offscreenDepthFormat);
    depthStencilView.setSubresourceRange({vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
    depthStencilView.setImage(image.image);

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                vk::ImageAspectFlagBits::eDepth);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass =
        nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                               true, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
  }

  // Creating the frame buffer for offscreen
  std::vector<vk::ImageView> attachments = {m_offscreenColor.descriptor.imageView,
                                            m_offscreenDepth.descriptor.imageView};

  m_device.destroy(m_offscreenFramebuffer);
  vk::FramebufferCreateInfo info;
  info.setRenderPass(m_offscreenRenderPass);
  info.setAttachmentCount(2);
  info.setPAttachments(attachments.data());
  info.setWidth(m_size.width);
  info.setHeight(m_size.height);
  info.setLayers(1);
  m_offscreenFramebuffer = m_device.createFramebuffer(info);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline()
{
  // Push constants in the fragment shader
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(float)};

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_postDescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_postPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout,
                                                            m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, defaultSearchPaths,
                                            true),
                              vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, defaultSearchPaths, true),
                              vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_postPipeline = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  m_postDescSetLayoutBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  vk::WriteDescriptorSet writeDescriptorSets =
      m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
  m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(vk::CommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  cmdBuf.pushConstants<float>(m_postPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0,
                              aspectRatio);
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_postPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_postPipelineLayout, 0,
                            m_postDescSet, {});
  cmdBuf.draw(3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
// #VKRay
void HelloVulkan::initRayTracing()
{
  // Requesting ray tracing properties
  auto properties =
      m_physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
                                      vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
}

//--------------------------------------------------------------------------------------------------
// Converting a OBJ primitive to the ray tracing geometry used for the BLAS
//
nvvk::RaytracingBuilderKHR::BlasInput HelloVulkan::objectToVkGeometryKHR(const ObjModel& model)
{
  vk::DeviceAddress vertexAddress = m_device.getBufferAddress({model.vertexBuffer.buffer});
  vk::DeviceAddress indexAddress  = m_device.getBufferAddress({model.indexBuffer.buffer});

  vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
  triangles.setVertexData(vertexAddress);
  triangles.setVertexStride(sizeof(VertexObj));
  triangles.setIndexType(vk::IndexType::eUint32);
  triangles.setIndexData(indexAddress);
  triangles.setTransformData({});
  triangles.setMaxVertex(model.nbVertices);

  // Setting up the build info of the acceleration
  vk::AccelerationStructureGeometryKHR asGeom;
  asGeom.setGeometryType(vk::GeometryTypeKHR::eTriangles);
  asGeom.setFlags(vk::GeometryFlagBitsKHR::eOpaque);
  asGeom.geometry.setTriangles(triangles);

  vk::AccelerationStructureBuildRangeInfoKHR offset;
  offset.setFirstVertex(0);
  offset.setPrimitiveCount(model.nbIndices / 3);  // Nb triangles
  offset.setPrimitiveOffset(0);
  offset.setTransformOffset(0);

  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);
  return input;
}

//--------------------------------------------------------------------------------------------------
// Returning the ray tracing geometry used for the BLAS, containing all spheres
//
nvvk::RaytracingBuilderKHR::BlasInput HelloVulkan::sphereToVkGeometryKHR()
{
  vk::DeviceAddress dataAddress = m_device.getBufferAddress({m_spheresAabbBuffer.buffer});

  vk::AccelerationStructureGeometryAabbsDataKHR aabbs;
  aabbs.setData(dataAddress);
  aabbs.setStride(sizeof(Aabb));

  // Setting up the build info of the acceleration (C version, c++ gives wrong type)
  vk::AccelerationStructureGeometryKHR asGeom(vk::GeometryTypeKHR::eAabbs, aabbs,
                                              vk::GeometryFlagBitsKHR::eOpaque);
  //asGeom.geometryType   = vk::GeometryTypeKHR::eAabbs;
  //asGeom.flags          = vk::GeometryFlagBitsKHR::eOpaque;
  //asGeom.geometry.aabbs = aabbs;


  vk::AccelerationStructureBuildRangeInfoKHR offset;
  offset.setFirstVertex(0);
  offset.setPrimitiveCount((uint32_t)m_spheres.size());  // Nb aabb
  offset.setPrimitiveOffset(0);
  offset.setTransformOffset(0);

  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);

  return input;
}

// Returning the ray tracing geometry used for the BLAS, containing all spheres
//
nvvk::RaytracingBuilderKHR::BlasInput HelloVulkan::bezierToVkGeometryKHR()
{
  vk::DeviceAddress dataAddress = m_device.getBufferAddress({m_bezierAabbBuffer.buffer});

  vk::AccelerationStructureGeometryAabbsDataKHR aabbs;
  aabbs.setData(dataAddress);
  aabbs.setStride(sizeof(Aabb));

  // Setting up the build info of the acceleration (C version, c++ gives wrong type)
  vk::AccelerationStructureGeometryKHR asGeom(vk::GeometryTypeKHR::eAabbs, aabbs,
                                              vk::GeometryFlagBitsKHR::eOpaque);
  //asGeom.geometryType   = vk::GeometryTypeKHR::eAabbs;
  //asGeom.flags          = vk::GeometryFlagBitsKHR::eOpaque;
  //asGeom.geometry.aabbs = aabbs;


  vk::AccelerationStructureBuildRangeInfoKHR offset;
  offset.setFirstVertex(0);
  offset.setPrimitiveCount((uint32_t)m_beziers.size());  // Nb aabb
  offset.setPrimitiveOffset(0);
  offset.setTransformOffset(0);

  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);
  return input;
}

// Returning the ray tracing geometry used for the BLAS, containing all spheres
//
nvvk::RaytracingBuilderKHR::BlasInput HelloVulkan::hairToVkGeometryKHR()
{
  vk::DeviceAddress dataAddress = m_device.getBufferAddress({m_hairsAabbBuffer.buffer});

  vk::AccelerationStructureGeometryAabbsDataKHR aabbs;
  aabbs.setData(dataAddress);
  aabbs.setStride(sizeof(Aabb));

  // Setting up the build info of the acceleration (C version, c++ gives wrong type)
  vk::AccelerationStructureGeometryKHR asGeom(vk::GeometryTypeKHR::eAabbs, aabbs,
                                              vk::GeometryFlagBitsKHR::eOpaque);
  //asGeom.geometryType   = vk::GeometryTypeKHR::eAabbs;
  //asGeom.flags          = vk::GeometryFlagBitsKHR::eOpaque;
  //asGeom.geometry.aabbs = aabbs;


  vk::AccelerationStructureBuildRangeInfoKHR offset;
  offset.setFirstVertex(0);
  offset.setPrimitiveCount((uint32_t)m_hairs.size());  // Nb aabb
  offset.setPrimitiveOffset(0);
  offset.setTransformOffset(0);

  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);
  return input;
}

//--------------------------------------------------------------------------------------------------
// Creating all spheres
//
void HelloVulkan::createSpheres(uint32_t nbSpheres)
{
#if 0
  std::random_device                    rd{};
  std::mt19937                          gen{rd()};
  std::normal_distribution<float>       xzd{0.f, 5.f};
  std::normal_distribution<float>       yd{6.f, 3.f};
  std::uniform_real_distribution<float> radd{.05f, .2f};

  // All spheres
  Sphere s;
  m_spheres.resize(nbSpheres);
  for(uint32_t i = 0; i < nbSpheres; i++)
  {
    s.center     = nvmath::vec3f(xzd(gen), yd(gen), xzd(gen));
    s.radius     = radd(gen);
    m_spheres[i] = std::move(s);
  }

  // Axis aligned bounding box of each sphere
  std::vector<Aabb> aabbs;
  aabbs.reserve(nbSpheres);
  for(const auto& s : m_spheres)
  {
    Aabb aabb;
    aabb.minimum = s.center - nvmath::vec3f(s.radius);
    aabb.maximum = s.center + nvmath::vec3f(s.radius);
    aabbs.emplace_back(aabb);
  }

  // Creating two materials
  MaterialObj mat;
  mat.diffuse = nvmath::vec3f(0, 1, 1);
  std::vector<MaterialObj> materials;
  std::vector<int>         matIdx(nbSpheres);
  materials.emplace_back(mat);
  mat.diffuse = nvmath::vec3f(1, 1, 0);
  materials.emplace_back(mat);

  // Assign a material to each sphere
  for(size_t i = 0; i < m_spheres.size(); i++)
  {
    matIdx[i] = i % 2;
  }

  // Creating all buffers
  using vkBU = vk::BufferUsageFlagBits;
  nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
  auto              cmdBuf = genCmdBuf.createCommandBuffer();
  m_spheresBuffer          = m_alloc.createBuffer(cmdBuf, m_spheres, vkBU::eStorageBuffer);
  m_spheresAabbBuffer      = m_alloc.createBuffer(cmdBuf, aabbs, vkBU::eShaderDeviceAddress);
  m_spheresMatIndexBuffer  = m_alloc.createBuffer(cmdBuf, matIdx, vkBU::eStorageBuffer);
  m_spheresMatColorBuffer  = m_alloc.createBuffer(cmdBuf, materials, vkBU::eStorageBuffer);
  genCmdBuf.submitAndWait(cmdBuf);

  // Debug information
  m_debug.setObjectName(m_spheresBuffer.buffer, "spheres");
  m_debug.setObjectName(m_spheresAabbBuffer.buffer, "spheresAabb");
  m_debug.setObjectName(m_spheresMatColorBuffer.buffer, "spheresMat");
  m_debug.setObjectName(m_spheresMatIndexBuffer.buffer, "spheresMatIdx");

#endif
}

// Creating all beziers
//

void HelloVulkan::createBeziers(uint32_t nbBeziers)
{
#if 0
  // All beziers
  m_beziers.push_back(Bezier{nvmath::vec3(15.0f, 1.0f, 15.0f), nvmath::vec3(10.0f, 10.0f, -5.0f),
                             nvmath::vec3(-10.0f, 15.0f, 5.0f), nvmath::vec3(-15.0f, 5.0f, -15.0f),
                             0.2f});
  // Creating two materials
  MaterialObj mat;
  mat.diffuse = nvmath::vec3f(0, 1, 1);
  std::vector<MaterialObj> materials;
  std::vector<int>         matIdx(nbBeziers);
  materials.emplace_back(mat);
  mat.diffuse = nvmath::vec3f(1, 1, 0);
  materials.emplace_back(mat);

  // Creating all buffers
  using vkBU = vk::BufferUsageFlagBits;
  nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
  auto              cmdBuf = genCmdBuf.createCommandBuffer();
  std::vector<Aabb> bezierAabbs;

  //Box Bezier
  for(const auto bezier : m_beziers)
  {
    // extremes
    nvmath::vec3 mi = nvmath::nv_min(bezier.p0, bezier.p3);
    nvmath::vec3 ma = nvmath::nv_max(bezier.p0, bezier.p3);

    // note pascal triangle coefficnets
    nvmath::vec3 c = -1.0f * bezier.p0 + 1.0f * bezier.p1;
    nvmath::vec3 b = 1.0f * bezier.p0 - 2.0f * bezier.p1 + 1.0f * bezier.p2;
    nvmath::vec3 a = -1.0f * bezier.p0 + 3.0f * bezier.p1 - 3.0f * bezier.p2 + 1.0f * bezier.p3;

    nvmath::vec3 h = b * b - a * c;

    // real solutions
    if(h.x > 0.0f || h.y > 0.0f || h.z > 0.0f)
    {
      nvmath::vec3 g = nvmath::nv_abs(h);
      g.x            = sqrt(g.x);
      g.y            = sqrt(g.y);
      g.z            = sqrt(g.z);

      nvmath::vec3 t1 = nvmath::nv_clamp((-b - g) / a, 0.0f, 1.0f);
      nvmath::vec3 s1 = nvmath::vec3(1.0f, 1.0f, 1.0f) - t1;
      nvmath::vec3 t2 = nvmath::nv_clamp((-b + g) / a, 0.0f, 1.0f);
      nvmath::vec3 s2 = nvmath::vec3(1.0f, 1.0f, 1.0f) - t2;
      nvmath::vec3 q1 = s1 * s1 * s1 * bezier.p0 + 3.0f * s1 * s1 * t1 * bezier.p1
                        + 3.0f * s1 * t1 * t1 * bezier.p2 + t1 * t1 * t1 * bezier.p3;
      nvmath::vec3 q2 = s2 * s2 * s2 * bezier.p0 + 3.0f * s2 * s2 * t2 * bezier.p1
                        + 3.0f * s2 * t2 * t2 * bezier.p2 + t2 * t2 * t2 * bezier.p3;

      if(h.x > 0.0)
      {
        mi.x = nvmath::nv_min(mi.x, nvmath::nv_min(q1.x, q2.x));
        ma.x = nvmath::nv_max(ma.x, nvmath::nv_max(q1.x, q2.x));
      }
      if(h.y > 0.0)
      {
        mi.y = nvmath::nv_min(mi.y, nvmath::nv_min(q1.y, q2.y));
        ma.y = nvmath::nv_max(ma.y, nvmath::nv_max(q1.y, q2.y));
      }
      if(h.z > 0.0)
      {
        mi.z = nvmath::nv_min(mi.z, nvmath::nv_min(q1.z, q2.z));
        ma.z = nvmath::nv_max(ma.z, nvmath::nv_max(q1.z, q2.z));
      }
    }
    Aabb aabb;
    aabb.minimum = mi;
    aabb.maximum = ma;
    aabb.minimum -= nvmath::vec3(bezier.thickness, bezier.thickness, bezier.thickness);
    aabb.maximum += nvmath::vec3(bezier.thickness, bezier.thickness, bezier.thickness);
    bezierAabbs.emplace_back(aabb);
  }

  m_beziersBuffer         = m_alloc.createBuffer(cmdBuf, m_beziers, vkBU::eStorageBuffer);
  m_bezierAabbBuffer      = m_alloc.createBuffer(cmdBuf, bezierAabbs, vkBU::eShaderDeviceAddress);
  m_bezierMatIndexBuffer  = m_alloc.createBuffer(cmdBuf, matIdx, vkBU::eStorageBuffer);
  m_beziersMatColorBuffer = m_alloc.createBuffer(cmdBuf, materials, vkBU::eStorageBuffer);
  genCmdBuf.submitAndWait(cmdBuf);

  // Debug information
  m_debug.setObjectName(m_beziersBuffer.buffer, "bezier");
  m_debug.setObjectName(m_bezierAabbBuffer.buffer, "bezierAabb");
  m_debug.setObjectName(m_beziersMatColorBuffer.buffer, "bezierMat");
  m_debug.setObjectName(m_bezierMatIndexBuffer.buffer, "bezierMatIdx");
	
#endif
}



//Create Hair


void HelloVulkan::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  allBlas.reserve(m_objModel.size());
  for(const auto& obj : m_objModel)
  {
    auto blas = objectToVkGeometryKHR(obj);

    // We could add more geometry in each BLAS, but we add only one for now
    //allBlas.emplace_back(blas);
  }

  // Spheres and bezier
  {
   // auto blas       = sphereToVkGeometryKHR();
    //auto blasBezier = bezierToVkGeometryKHR();
    auto blasHair   = hairToVkGeometryKHR();
   // allBlas.emplace_back(blas);
   // allBlas.emplace_back(blasBezier);
    allBlas.emplace_back(blasHair);
  }

  m_rtBuilder.buildBlas(allBlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
}

void HelloVulkan::createTopLevelAS()
{
  std::vector<nvvk::RaytracingBuilderKHR::Instance> tlas;
  tlas.reserve(m_objInstance.size());
  /*for(uint32_t i = 0; i < static_cast<uint32_t>(m_objInstance.size()); i++)
  {
    nvvk::RaytracingBuilderKHR::Instance rayInst;
    rayInst.transform        = m_objInstance[i].transform;  // Position of the instance
    rayInst.instanceCustomId = i;                           // gl_InstanceCustomIndexEXT
    rayInst.blasId           = m_objInstance[i].objIndex;
    rayInst.hitGroupId       = 0;  // We will use the same hit group for all objects
    rayInst.flags            = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    tlas.emplace_back(rayInst);
  }*/

  // Add the blas containing all spheres, beziers, hairs
#if 0
  {
    nvvk::RaytracingBuilderKHR::Instance rayInst;
    rayInst.transform        = m_objInstance[0].transform;          // Position of the instance
    rayInst.instanceCustomId = static_cast<uint32_t>(tlas.size());  // gl_InstanceCustomIndexEXT
    rayInst.blasId           = 1;  //static_cast<uint32_t>(m_objModel.size());
    rayInst.hitGroupId       = 1;  // We will use the same hit group for all objects
    rayInst.flags            = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    tlas.emplace_back(rayInst);
  }

  {
    nvvk::RaytracingBuilderKHR::Instance rayInst;
    rayInst.transform        = m_objInstance[0].transform;          // Position of the instance
    rayInst.instanceCustomId = static_cast<uint32_t>(tlas.size());  // gl_InstanceCustomIndexEXT
    rayInst.blasId           = 2;
    rayInst.hitGroupId       = 1;  // We will use the same hit group for all objects
    rayInst.flags            = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    tlas.emplace_back(rayInst);
  }
#endif
#if 1
  {
    nvvk::RaytracingBuilderKHR::Instance rayInst;
    rayInst.transform        = m_objInstance[0].transform;          // Position of the instance
    rayInst.instanceCustomId = static_cast<uint32_t>(tlas.size());  // gl_InstanceCustomIndexEXT
    rayInst.blasId           = 0;
    rayInst.hitGroupId       = 1;  // We will use the same hit group for all objects
    rayInst.flags            = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    tlas.emplace_back(rayInst);
  }
#endif
  m_rtBuilder.buildTlas(tlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void HelloVulkan::createRtDescriptorSet()
{
  using vkDT   = vk::DescriptorType;
  using vkSS   = vk::ShaderStageFlagBits;
  using vkDSLB = vk::DescriptorSetLayoutBinding;

  m_rtDescSetLayoutBind.addBinding(vkDSLB(0, vkDT::eAccelerationStructureKHR, 1,
                                          vkSS::eRaygenKHR | vkSS::eClosestHitKHR));  // TLAS
  m_rtDescSetLayoutBind.addBinding(
      vkDSLB(1, vkDT::eStorageImage, 1, vkSS::eRaygenKHR));  // Output image

  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);
  m_rtDescSet       = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::AccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  vk::WriteDescriptorSetAccelerationStructureKHR descASInfo;
  descASInfo.setAccelerationStructureCount(1);
  descASInfo.setPAccelerationStructures(&tlas);
  vk::DescriptorImageInfo imageInfo{
      {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 0, &descASInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 1, &imageInfo));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void HelloVulkan::updateRtDescriptorSet()
{
  using vkDT = vk::DescriptorType;

  // (1) Output buffer
  vk::DescriptorImageInfo imageInfo{
      {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};
  vk::WriteDescriptorSet wds{m_rtDescSet, 1, 0, 1, vkDT::eStorageImage, &imageInfo};
  m_device.updateDescriptorSets(wds, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void HelloVulkan::createRtPipeline()
{
  vk::ShaderModule raygenSM = nvvk::createShaderModule(
      m_device, nvh::loadFile("spv/raytrace.rgen.spv", true, defaultSearchPaths, true));
  vk::ShaderModule missSM = nvvk::createShaderModule(
      m_device, nvh::loadFile("spv/raytrace.rmiss.spv", true, defaultSearchPaths, true));

  // The second miss shader is invoked when a shadow ray misses the geometry. It
  // simply indicates that no occlusion has been found
  vk::ShaderModule shadowmissSM = nvvk::createShaderModule(
      m_device, nvh::loadFile("spv/raytraceShadow.rmiss.spv", true, defaultSearchPaths, true));


  std::vector<vk::PipelineShaderStageCreateInfo> stages;

  // Raygen
  vk::RayTracingShaderGroupCreateInfoKHR rg{vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                            static_cast<uint32_t>(stages.size()),
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR};
  //rg.setGeneralShader(static_cast<uint32_t>(stages.size()));
  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenKHR, raygenSM, "main"});
  m_rtShaderGroups.push_back(rg);
  // Miss
  vk::RayTracingShaderGroupCreateInfoKHR mg{vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
  mg.setGeneralShader(static_cast<uint32_t>(stages.size()));
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, missSM, "main"});
  m_rtShaderGroups.push_back(mg);
  // Shadow Miss
  mg.setGeneralShader(static_cast<uint32_t>(stages.size()));
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, shadowmissSM, "main"});
  m_rtShaderGroups.push_back(mg);

  // Hit Group0 - Closest Hit
  vk::ShaderModule chitSM = nvvk::createShaderModule(
      m_device, nvh::loadFile("spv/raytrace.rchit.spv", true, defaultSearchPaths, true));

  {
    vk::RayTracingShaderGroupCreateInfoKHR hg{vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                                              VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                              VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    hg.setClosestHitShader(static_cast<uint32_t>(stages.size()));
    stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, chitSM, "main"});
    m_rtShaderGroups.push_back(hg);
  }

  // Hit Group1 - Closest Hit + Intersection (procedural)
  vk::ShaderModule chit2SM = nvvk::createShaderModule(
      m_device, nvh::loadFile("spv/raytrace2.rchit.spv", true, defaultSearchPaths, true));
  vk::ShaderModule rintSM = nvvk::createShaderModule(
      m_device, nvh::loadFile("spv/raytrace.rint.spv", true, defaultSearchPaths, true));
  {
    vk::RayTracingShaderGroupCreateInfoKHR hg{vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup,
                                              VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                              VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    hg.setClosestHitShader(static_cast<uint32_t>(stages.size()));
    stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, chit2SM, "main"});
    hg.setIntersectionShader(static_cast<uint32_t>(stages.size()));
    stages.push_back({{}, vk::ShaderStageFlagBits::eIntersectionKHR, rintSM, "main"});
    m_rtShaderGroups.push_back(hg);
  }

  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

  // Push constant: we want to be able to update constants used by the shaders
  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenKHR
                                         | vk::ShaderStageFlagBits::eClosestHitKHR
                                         | vk::ShaderStageFlagBits::eMissKHR,
                                     0, sizeof(RtPushConstant)};
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<vk::DescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(rtDescSetLayouts.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(rtDescSetLayouts.data());

  m_rtPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  vk::RayTracingPipelineCreateInfoKHR rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));  // Stages are shaders
  rayPipelineInfo.setPStages(stages.data());

  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(
      m_rtShaderGroups.size()));  // 1-raygen, n-miss, n-(hit[+anyhit+intersect])
  rayPipelineInfo.setPGroups(m_rtShaderGroups.data());

  rayPipelineInfo.setMaxPipelineRayRecursionDepth(2);  // Ray depth
  rayPipelineInfo.setLayout(m_rtPipelineLayout);
  m_rtPipeline = static_cast<const vk::Pipeline&>(
      m_device.createRayTracingPipelineKHR({}, {}, rayPipelineInfo));

  m_device.destroy(raygenSM);
  m_device.destroy(missSM);
  m_device.destroy(shadowmissSM);
  m_device.destroy(chitSM);
  m_device.destroy(chit2SM);
  m_device.destroy(rintSM);
}

//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and writing them in a SBT buffer
// - Besides exception, this could be always done like this
//   See how the SBT buffer is used in run()
//
void HelloVulkan::createRtShaderBindingTable()
{
  auto groupCount =
      static_cast<uint32_t>(m_rtShaderGroups.size());  // shaders: raygen, 2 miss, 2 chit, rint
  uint32_t groupHandleSize = m_rtProperties.shaderGroupHandleSize;  // Size of a program identifier
  uint32_t groupSizeAligned =
      nvh::align_up(groupHandleSize, m_rtProperties.shaderGroupBaseAlignment);

  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t sbtSize = groupCount * groupSizeAligned;

  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  auto result = m_device.getRayTracingShaderGroupHandlesKHR(m_rtPipeline, 0, groupCount, sbtSize,
                                                            shaderHandleStorage.data());
  assert(result == vk::Result::eSuccess);

  // Write the handles in the SBT
  m_rtSBTBuffer = m_alloc.createBuffer(
      sbtSize,
      vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddressKHR
          | vk::BufferUsageFlagBits::eShaderBindingTableKHR,
      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  m_debug.setObjectName(m_rtSBTBuffer.buffer, std::string("SBT").c_str());

  // Write the handles in the SBT
  void* mapped = m_alloc.map(m_rtSBTBuffer);
  auto* pData  = reinterpret_cast<uint8_t*>(mapped);
  for(uint32_t g = 0; g < groupCount; g++)
  {
    memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
    pData += groupSizeAligned;
  }
  m_alloc.unmap(m_rtSBTBuffer);


  m_alloc.finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void HelloVulkan::raytrace(const vk::CommandBuffer& cmdBuf, const nvmath::vec4f& clearColor)
{
  m_debug.beginLabel(cmdBuf, "Ray trace");
  // Initializing push constant values
  m_rtPushConstants.clearColor     = clearColor;
  m_rtPushConstants.lightPosition  = m_pushConstant.lightPosition;
  m_rtPushConstants.lightIntensity = m_pushConstant.lightIntensity;
  m_rtPushConstants.lightType      = m_pushConstant.lightType;

  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipelineLayout, 0,
                            {m_rtDescSet, m_descSet}, {});
  cmdBuf.pushConstants<RtPushConstant>(m_rtPipelineLayout,
                                       vk::ShaderStageFlagBits::eRaygenKHR
                                           | vk::ShaderStageFlagBits::eClosestHitKHR
                                           | vk::ShaderStageFlagBits::eMissKHR,
                                       0, m_rtPushConstants);

  // Size of a program identifier
  uint32_t groupSize =
      nvh::align_up(m_rtProperties.shaderGroupHandleSize, m_rtProperties.shaderGroupBaseAlignment);
  uint32_t          groupStride = groupSize;
  vk::DeviceAddress sbtAddress  = m_device.getBufferAddress({m_rtSBTBuffer.buffer});

  using Stride = vk::StridedDeviceAddressRegionKHR;
  std::array<Stride, 4> strideAddresses{
      Stride{sbtAddress + 0u * groupSize, groupStride, groupSize * 1},  // raygen
      Stride{sbtAddress + 1u * groupSize, groupStride, groupSize * 2},  // miss
      Stride{sbtAddress + 3u * groupSize, groupStride, groupSize * 1},  // hit
      Stride{0u, 0u, 0u}};                                              // callable

  cmdBuf.traceRaysKHR(&strideAddresses[0], &strideAddresses[1], &strideAddresses[2],
                      &strideAddresses[3],              //
                      m_size.width, m_size.height, 1);  //

  m_debug.endLabel(cmdBuf);
}
