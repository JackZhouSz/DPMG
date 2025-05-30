//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// MetalFX/MTLFXDefines.hpp
//
// Copyright 2020-2023 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "../Foundation/NSDefines.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#define _MTLFX_EXPORT                           _NS_EXPORT
#define _MTLFX_EXTERN                           _NS_EXTERN
#define _MTLFX_INLINE                           _NS_INLINE
#define _MTLFX_PACKED                           _NS_PACKED

#define _MTLFX_CONST( type, name )              _NS_CONST( type, name )
#define _MTLFX_ENUM( type, name )               _NS_ENUM( type, name )
#define _MTLFX_OPTIONS( type, name )            _NS_OPTIONS( type, name )

#define _MTLFX_VALIDATE_SIZE( mtlfx, name )     _NS_VALIDATE_SIZE( mtlfx, name )
#define _MTLFX_VALIDATE_ENUM( mtlfx, name )     _NS_VALIDATE_ENUM( mtlfx, name )

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
