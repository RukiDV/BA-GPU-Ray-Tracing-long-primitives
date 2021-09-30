/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "raycommon.glsl"
#include "wavefront.glsl"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(buffer_reference, scalar) buffer Vertices { Vertex v[]; };// Positions of an object
layout(buffer_reference, scalar) buffer Indices { ivec3 i[]; };// Triangle indices
layout(buffer_reference, scalar) buffer Materials { WaveFrontMaterial m[]; };// Array of all materials on an object
layout(buffer_reference, scalar) buffer MatIndices { int i[]; };// Material ID for each triangle
layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 1, scalar) buffer SceneDesc_ { SceneDesc i[]; } sceneDesc;
layout(binding = 2, set = 1) uniform sampler2D textureSamplers[];
layout(binding = 3, set = 1, scalar, std430) buffer allHairs_ { Hair allHairs[]; };
// clang-format on

layout(push_constant) uniform Constants
{
    vec4  clearColor;
    vec3  lightPosition;
    float lightIntensity;
    int   lightType;
}
pushC;


void main()
{
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    if (gl_HitTEXT < 9999)
    {
        Hair instanceHair = allHairs[gl_PrimitiveID];

        vec3 normal = instanceHair.v1.n;
        vec3 color = instanceHair.v0.c;
        // Vector toward the light
        vec3  L;
        float lightIntensity = pushC.lightIntensity;
        float lightDistance  = 100000.0;
        // Point light
        if (pushC.lightType == 0)
        {
            vec3 lDir      = pushC.lightPosition - worldPos;
            lightDistance  = length(lDir);
            lightIntensity = pushC.lightIntensity / (lightDistance * lightDistance);
            L              = normalize(lDir);
        }
        else // Directional light
        {
            L = normalize(pushC.lightPosition - vec3(0));
        }

        // Diffuse
        vec3  diffuse     = computeDiffuse(color, L, normal);
        vec3  specular    = vec3(0);
        float attenuation = 0.3;

        // Tracing shadow ray only if the light is visible from the surface
        if (dot(normal, L) > 0)
        {
            float tMin   = 0.001;
            float tMax   = 100.0;
            vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
            vec3  rayDir = L;
            uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT
            | gl_RayFlagsSkipClosestHitShaderEXT;
            isShadowed = true;
            traceRayEXT(topLevelAS, // acceleration structure
            flags, // rayFlags
            0xFF, // cullMask
            0, // sbtRecordOffset
            0, // sbtRecordStride
            1, // missIndex
            origin, // ray origin
            tMin, // ray min range
            rayDir, // ray direction
            tMax, // ray max range
            1// payload (location = 1)
            );

            if (isShadowed)
            {
                attenuation = 0.3;
            }
            else
            {
                attenuation = 1;
                // Specular
                //specular = computeSpecular(gl_WorldRayDirectionEXT, L, normal);
            }
        }

        float randR = (gl_InstanceID % 675 + 6);
        float randG = (gl_InstanceID % 245 + 4);
        float randB = (gl_InstanceID % 558 + 8);
        randR = fract(sin(dot(vec2(randG, randB), vec2(12.9898, 78.233))) * 43758.5453);
        randG = fract(sin(dot(vec2(randR, randG), vec2(12.9898, 78.233))) * 43758.5453);
        randB = fract(sin(dot(vec2(randB, randR), vec2(12.9898, 78.233))) * 43758.5453);
        prd.hitValue = vec3(lightIntensity * (attenuation * color + diffuse));
    }
    else
    {
        prd.hitValue = vec3(0.0, 0.0, 0.0);
    }
}