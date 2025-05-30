use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::ffi::{c_char, c_void};
pub const INVALID_RESOURCE_HANDLE: u64 = u64::MAX;
pub type DispatchCallback = extern "C" fn(*mut u8);
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct CreatedResourceInfo {
    pub handle: u64,
    pub native_handle: *mut c_void,
}
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct CreatedResourceInfoRemote {
    pub handle: u64,
}
impl CreatedResourceInfo {
    pub const INVALID: Self = Self {
        handle: INVALID_RESOURCE_HANDLE,
        native_handle: std::ptr::null_mut(),
    };
    #[inline]
    pub fn valid(&self) -> bool {
        self.handle != INVALID_RESOURCE_HANDLE
    }
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct CreatedBufferInfoRemote {
    pub resource: CreatedResourceInfoRemote,
    pub element_stride: usize,
    pub total_size_bytes: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct CreatedSwapchainInfoRemote {
    pub resource: CreatedResourceInfoRemote,
    pub storage: PixelStorage,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct CreatedBufferInfo {
    pub resource: CreatedResourceInfo,
    pub element_stride: usize,
    pub total_size_bytes: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct CreatedSwapchainInfo {
    pub resource: CreatedResourceInfo,
    pub storage: PixelStorage,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct CreatedShaderInfo {
    pub resource: CreatedResourceInfo,
    pub block_size: [u32; 3],
}
unsafe impl Send for CreatedShaderInfo {}
unsafe impl Sync for CreatedShaderInfo {}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct ShaderOption {
    pub enable_cache: bool,
    pub enable_fast_math: bool,
    pub enable_debug_info: bool,
    pub compile_only: bool,
    pub time_trace: bool,
    pub max_registers: u32,
    pub name: *const std::ffi::c_char,
    pub native_include: *const std::ffi::c_char,
}
unsafe impl Send for ShaderOption {}
unsafe impl Sync for ShaderOption {}
impl Default for ShaderOption {
    fn default() -> Self {
        Self {
            enable_cache: true,
            enable_fast_math: true,
            enable_debug_info: false,
            compile_only: false,
            time_trace: false,
            max_registers: 0,
            name: std::ptr::null(),
            native_include: std::ptr::null(),
        }
    }
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Buffer(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Context(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Device(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Texture(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Stream(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Event(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Swapchain(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct BindlessArray(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Mesh(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Curve(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct ProceduralPrimitive(pub u64);
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Accel(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct IrModule(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Shader(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct NodeRef(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub enum AccelUsageHint {
    FastTrace,
    FastBuild,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub enum AccelBuildRequest {
    PreferUpdate,
    ForceBuild,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub enum CurveBasis {
    PiecewiseLinear = 0,
    CubicBSpline = 1,
    CatmullRom = 2,
    Bezier = 3,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct AccelOption {
    pub hint: AccelUsageHint,
    pub allow_compaction: bool,
    pub allow_update: bool,
}
impl Default for AccelOption {
    fn default() -> Self {
        Self {
            hint: AccelUsageHint::FastTrace,
            allow_compaction: true,
            allow_update: false,
        }
    }
}

bitflags! {
    #[repr(C)]
    #[derive(Copy, Clone, Debug, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct AccelBuildModificationFlags : u32 {
        const EMPTY = 0;
        const PRIMITIVE = 1 << 0;
        const TRANSFORM = 1 << 1;
        const OPAQUE_ON = 1 << 2;
        const OPAQUE_OFF = 1 << 3;
        const OPAQUE = Self::OPAQUE_ON.bits() | Self::OPAQUE_OFF.bits();
        const VISIBILITY = 1 << 4;
        const USER_ID = 1 << 5;
    }
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum MeshType {
    Mesh,
    ProceduralPrimitive,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct AccelBuildModification {
    pub index: u32,
    pub user_id: u32,
    pub flags: AccelBuildModificationFlags,
    pub visibility: u32,
    pub mesh: u64,
    pub affine: [f32; 12],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub enum PixelStorage {
    Byte1,
    Byte2,
    Byte4,
    Short1,
    Short2,
    Short4,
    Int1,
    Int2,
    Int4,
    Half1,
    Half2,
    Half4,
    Float1,
    Float2,
    Float4,

    R10G10B10A2,
    R11G11B10,

    Bc1,
    Bc2,
    Bc3,
    Bc4,
    Bc5,
    Bc6,
    Bc7,
}

impl PixelStorage {
    pub fn size(&self) -> usize {
        match self {
            PixelStorage::Byte1 => 1,
            PixelStorage::Byte2 => 2,
            PixelStorage::Byte4 => 4,
            PixelStorage::Short1 => 2,
            PixelStorage::Short2 => 4,
            PixelStorage::Short4 => 8,
            PixelStorage::Int1 => 4,
            PixelStorage::Int2 => 8,
            PixelStorage::Int4 => 16,
            PixelStorage::Half1 => 2,
            PixelStorage::Half2 => 4,
            PixelStorage::Half4 => 8,
            PixelStorage::Float1 => 4,
            PixelStorage::Float2 => 8,
            PixelStorage::Float4 => 16,
            _ => todo!(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub enum PixelFormat {
    R8Sint,
    R8Uint,
    R8Unorm,

    Rg8Sint,
    Rg8Uint,
    Rg8Unorm,

    Rgba8Sint,
    Rgba8Uint,
    Rgba8Unorm,

    R16Sint,
    R16Uint,
    R16Unorm,

    Rg16Sint,
    Rg16Uint,
    Rg16Unorm,

    Rgba16Sint,
    Rgba16Uint,
    Rgba16Unorm,

    R32Sint,
    R32Uint,

    Rg32Sint,
    Rg32Uint,

    Rgba32Sint,
    Rgba32Uint,

    R16f,
    Rg16f,
    Rgba16f,

    R32f,
    Rg32f,
    Rgba32f,

    R10G10B10A2UInt,
    R10G10B10A2UNorm,
    R11G11B10F,

    BC1UNorm,
    BC2UNorm,
    BC3UNorm,
    BC4UNorm,
    BC5UNorm,
    BC6HUF16,
    BC7UNorm,
}
impl PixelFormat {
    pub fn storage(&self) -> PixelStorage {
        match self {
            PixelFormat::R8Sint | PixelFormat::R8Uint | PixelFormat::R8Unorm => PixelStorage::Byte1,
            PixelFormat::Rg8Sint | PixelFormat::Rg8Uint | PixelFormat::Rg8Unorm => {
                PixelStorage::Byte2
            }
            PixelFormat::Rgba8Sint | PixelFormat::Rgba8Uint | PixelFormat::Rgba8Unorm => {
                PixelStorage::Byte4
            }
            PixelFormat::R16Sint | PixelFormat::R16Uint | PixelFormat::R16Unorm => {
                PixelStorage::Short1
            }
            PixelFormat::Rg16Sint | PixelFormat::Rg16Uint | PixelFormat::Rg16Unorm => {
                PixelStorage::Short2
            }
            PixelFormat::Rgba16Sint | PixelFormat::Rgba16Uint | PixelFormat::Rgba16Unorm => {
                PixelStorage::Short4
            }
            PixelFormat::R32Sint | PixelFormat::R32Uint => PixelStorage::Int1,
            PixelFormat::Rg32Sint | PixelFormat::Rg32Uint => PixelStorage::Int2,
            PixelFormat::R16f => PixelStorage::Half1,
            PixelFormat::Rg16f => PixelStorage::Half2,
            PixelFormat::Rgba16f => PixelStorage::Half4,
            PixelFormat::R32f => PixelStorage::Float1,
            PixelFormat::Rg32f => PixelStorage::Float2,
            PixelFormat::Rgba32Sint | PixelFormat::Rgba32Uint => PixelStorage::Int4,
            PixelFormat::Rgba32f => PixelStorage::Float4,
            PixelFormat::BC1UNorm => PixelStorage::Bc1,
            PixelFormat::BC2UNorm => PixelStorage::Bc2,
            PixelFormat::BC3UNorm => PixelStorage::Bc3,
            PixelFormat::BC4UNorm => PixelStorage::Bc4,
            PixelFormat::BC5UNorm => PixelStorage::Bc5,
            PixelFormat::BC6HUF16 => PixelStorage::Bc6,
            PixelFormat::BC7UNorm => PixelStorage::Bc7,
            PixelFormat::R10G10B10A2UInt | PixelFormat::R10G10B10A2UNorm => {
                PixelStorage::R10G10B10A2
            }
            PixelFormat::R11G11B10F => PixelStorage::R11G11B10,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub enum SamplerFilter {
    Point,
    LinearPoint,
    LinearLinear,
    Anisotropic,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub enum SamplerAddress {
    Edge,
    Repeat,
    Mirror,
    Zero,
}

#[repr(C)]
#[derive(
    Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Default, Serialize, Deserialize,
)]
pub struct Sampler {
    pub filter: SamplerFilter,
    pub address: SamplerAddress,
}
impl Default for SamplerFilter {
    fn default() -> Self {
        SamplerFilter::Point
    }
}
impl Default for SamplerAddress {
    fn default() -> Self {
        SamplerAddress::Edge
    }
}
impl Sampler {
    pub fn encode(&self) -> u8 {
        (self.filter as u8) | ((self.address as u8) << 2)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferArgument {
    pub buffer: Buffer,
    pub offset: usize,
    pub size: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct TextureArgument {
    pub texture: Texture,
    pub level: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct UniformArgument {
    pub data: *const u8,
    pub size: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum Argument {
    Buffer(BufferArgument),
    Texture(TextureArgument),
    Uniform(UniformArgument),
    BindlessArray(BindlessArray),
    Accel(Accel),
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Capture {
    pub node: NodeRef,
    pub arg: Argument,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct KernelModule {
    pub ptr: u64, // pub ir_module: IrModule,
                  // pub captured: *const Capture,
                  // pub captured_count: usize,
                  // pub args: *const NodeRef,
                  // pub args_count: usize,
                  // pub shared: *const NodeRef,
                  // pub shared_count: usize,
                  // pub block_size: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct CallableModule {
    pub ir_module: IrModule,
    pub args: *const NodeRef,
    pub args_count: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferUploadCommand {
    pub buffer: Buffer,
    pub offset: usize,
    pub size: usize,
    pub data: *const u8,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferDownloadCommand {
    pub buffer: Buffer,
    pub offset: usize,
    pub size: usize,
    pub data: *mut u8,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferCopyCommand {
    pub src: Buffer,
    pub src_offset: usize,
    pub dst: Buffer,
    pub dst_offset: usize,
    pub size: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferToTextureCopyCommand {
    pub buffer: Buffer,
    pub buffer_offset: usize,
    pub texture: Texture,
    pub storage: PixelStorage,
    pub texture_level: u32,
    pub texture_size: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct TextureToBufferCopyCommand {
    pub buffer: Buffer,
    pub buffer_offset: usize,
    pub texture: Texture,
    pub storage: PixelStorage,
    pub texture_level: u32,
    pub texture_size: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct TextureUploadCommand {
    pub texture: Texture,
    pub storage: PixelStorage,
    pub level: u32,
    pub size: [u32; 3],
    pub data: *const u8,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct TextureDownloadCommand {
    pub texture: Texture,
    pub storage: PixelStorage,
    pub level: u32,
    pub size: [u32; 3],
    pub data: *mut u8,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct TextureCopyCommand {
    pub storage: PixelStorage,
    pub src: Texture,
    pub dst: Texture,
    pub size: [u32; 3],
    pub src_level: u32,
    pub dst_level: u32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct ShaderDispatchCommand {
    pub shader: Shader,
    pub dispatch_size: [u32; 3],
    pub args: *const Argument,
    pub args_count: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct MeshBuildCommand {
    pub mesh: Mesh,
    pub request: AccelBuildRequest,
    pub vertex_buffer: Buffer,
    pub vertex_buffer_offset: usize,
    pub vertex_buffer_size: usize,
    pub vertex_stride: usize,
    pub index_buffer: Buffer,
    pub index_buffer_offset: usize,
    pub index_buffer_size: usize,
    pub index_stride: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct CurveBuildCommand {
    pub curve: Curve,
    pub request: AccelBuildRequest,
    pub basis: CurveBasis,
    pub cp_count: usize,
    pub seg_count: usize,
    pub cp_buffer: Buffer,
    pub cp_buffer_offset: usize,
    pub cp_buffer_stride: usize,
    pub seg_buffer: Buffer,
    pub seg_buffer_offset: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct ProceduralPrimitiveBuildCommand {
    pub handle: ProceduralPrimitive,
    pub request: AccelBuildRequest,
    pub aabb_buffer: Buffer,
    pub aabb_buffer_offset: usize,
    pub aabb_count: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct AccelBuildCommand {
    pub accel: Accel,
    pub request: AccelBuildRequest,
    pub instance_count: u32,
    pub modifications: *const AccelBuildModification,
    pub modifications_count: usize,
    pub update_instance_buffer_only: bool,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum BindlessArrayUpdateOperation {
    None,
    Emplace,
    Remove,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BindlessArrayUpdateBuffer {
    pub op: BindlessArrayUpdateOperation,
    pub handle: Buffer,
    pub offset: usize,
}
impl Default for BindlessArrayUpdateBuffer {
    fn default() -> Self {
        Self {
            op: BindlessArrayUpdateOperation::None,
            handle: Buffer(INVALID_RESOURCE_HANDLE),
            offset: 0,
        }
    }
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BindlessArrayUpdateTexture {
    pub op: BindlessArrayUpdateOperation,
    pub handle: Texture,
    pub sampler: Sampler,
}
impl Default for BindlessArrayUpdateTexture {
    fn default() -> Self {
        Self {
            op: BindlessArrayUpdateOperation::None,
            handle: Texture(INVALID_RESOURCE_HANDLE),
            sampler: Sampler::default(),
        }
    }
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Default)]
pub struct BindlessArrayUpdateModification {
    pub slot: usize,
    pub buffer: BindlessArrayUpdateBuffer,
    pub tex2d: BindlessArrayUpdateTexture,
    pub tex3d: BindlessArrayUpdateTexture,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BindlessArrayUpdateCommand {
    pub handle: BindlessArray,
    pub modifications: *const BindlessArrayUpdateModification,
    pub modifications_count: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum Command {
    BufferUpload(BufferUploadCommand),
    BufferDownload(BufferDownloadCommand),
    BufferCopy(BufferCopyCommand),
    BufferToTextureCopy(BufferToTextureCopyCommand),
    TextureToBufferCopy(TextureToBufferCopyCommand),
    TextureUpload(TextureUploadCommand),
    TextureDownload(TextureDownloadCommand),
    TextureCopy(TextureCopyCommand),
    ShaderDispatch(ShaderDispatchCommand),
    MeshBuild(MeshBuildCommand),
    CurveBuild(CurveBuildCommand),
    ProceduralPrimitiveBuild(ProceduralPrimitiveBuildCommand),
    AccelBuild(AccelBuildCommand),
    BindlessArrayUpdate(BindlessArrayUpdateCommand),
}

unsafe impl Send for Command {}

unsafe impl Sync for Command {}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct CommandList {
    pub commands: *const Command,
    pub commands_count: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash, Serialize, Deserialize)]
pub enum StreamTag {
    Graphics,
    Compute,
    Copy,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LoggerMessage {
    pub target: *const c_char,
    pub level: *const c_char,
    pub message: *const c_char,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct LibInterface {
    pub inner: *mut c_void,
    pub set_logger_callback: unsafe extern "C" fn(unsafe extern "C" fn(LoggerMessage)),
    pub create_context: unsafe extern "C" fn(*const c_char) -> Context,
    pub destroy_context: unsafe extern "C" fn(Context),
    pub create_device:
        unsafe extern "C" fn(Context, *const c_char, *const c_char) -> DeviceInterface,
    pub free_string: unsafe extern "C" fn(*mut c_char),
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct DeviceInterface {
    pub device: Device,
    pub destroy_device: unsafe extern "C" fn(DeviceInterface),
    pub create_buffer:
        unsafe extern "C" fn(Device, *const c_void, usize, *mut c_void) -> CreatedBufferInfo,
    pub destroy_buffer: unsafe extern "C" fn(Device, Buffer),
    pub create_texture: unsafe extern "C" fn(
        Device,
        PixelFormat,
        u32,
        u32,
        u32,
        u32,
        u32,
        bool,
    ) -> CreatedResourceInfo,
    pub native_handle: unsafe extern "C" fn(Device) -> *mut c_void,
    pub compute_warp_size: unsafe extern "C" fn(Device) -> u32,
    pub destroy_texture: unsafe extern "C" fn(Device, Texture),
    pub create_bindless_array: unsafe extern "C" fn(Device, usize) -> CreatedResourceInfo,
    pub destroy_bindless_array: unsafe extern "C" fn(Device, BindlessArray),
    pub create_stream: unsafe extern "C" fn(Device, StreamTag) -> CreatedResourceInfo,
    pub destroy_stream: unsafe extern "C" fn(Device, Stream),
    pub synchronize_stream: unsafe extern "C" fn(Device, Stream),
    pub dispatch: unsafe extern "C" fn(Device, Stream, CommandList, DispatchCallback, *mut u8),
    pub create_swapchain: unsafe extern "C" fn(
        Device,
        u64,
        Stream,
        u32,
        u32,
        bool,
        bool,
        u32,
    ) -> CreatedSwapchainInfo,
    pub present_display_in_stream: unsafe extern "C" fn(Device, Stream, Swapchain, Texture),
    pub destroy_swapchain: unsafe extern "C" fn(Device, Swapchain),
    pub create_shader:
        unsafe extern "C" fn(Device, KernelModule, &ShaderOption) -> CreatedShaderInfo,
    pub destroy_shader: unsafe extern "C" fn(Device, Shader),
    pub create_event: unsafe extern "C" fn(Device) -> CreatedResourceInfo,
    pub destroy_event: unsafe extern "C" fn(Device, Event),
    pub signal_event: unsafe extern "C" fn(Device, Event, Stream, u64),
    pub synchronize_event: unsafe extern "C" fn(Device, Event, u64),
    pub wait_event: unsafe extern "C" fn(Device, Event, Stream, u64),
    pub is_event_completed: unsafe extern "C" fn(Device, Event, u64) -> bool,
    pub create_mesh: unsafe extern "C" fn(Device, &AccelOption) -> CreatedResourceInfo,
    pub destroy_mesh: unsafe extern "C" fn(Device, Mesh),
    pub create_curve: unsafe extern "C" fn(Device, &AccelOption) -> CreatedResourceInfo,
    pub destroy_curve: unsafe extern "C" fn(Device, Curve),
    pub create_procedural_primitive:
        unsafe extern "C" fn(Device, &AccelOption) -> CreatedResourceInfo,
    pub destroy_procedural_primitive: unsafe extern "C" fn(Device, ProceduralPrimitive),
    pub create_accel: unsafe extern "C" fn(Device, &AccelOption) -> CreatedResourceInfo,
    pub destroy_accel: unsafe extern "C" fn(Device, Accel),
    pub query: unsafe extern "C" fn(Device, *const c_char) -> *mut c_char,
    pub pinned_memory_ext: unsafe extern "C" fn(Device) -> PinnedMemoryExt,
    pub denoiser_ext: unsafe extern "C" fn(Device) -> DenoiserExt,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct StringView {
    pub data: *const c_char,
    pub size: usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct StringViewMut {
    pub data: *mut c_char,
    pub size: usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct BinaryIo {
    pub dtor: unsafe extern "C" fn(*mut BinaryIo),

    pub read_shader_bytecode: unsafe extern "C" fn(*mut BinaryIo, name: StringView) -> ByteStream,
    pub read_shader_cache: unsafe extern "C" fn(*mut BinaryIo, name: StringView) -> ByteStream,
    pub read_internal_shader_cache:
        unsafe extern "C" fn(*mut BinaryIo, name: StringView) -> ByteStream,

    pub write_shader_byte_code: unsafe extern "C" fn(
        *mut BinaryIo,
        name: StringView,
        data: *const u8,
        size: usize,
        out_path: StringViewMut,
    ),
    pub write_shader_cache: unsafe extern "C" fn(
        *mut BinaryIo,
        name: StringView,
        data: *const u8,
        size: usize,
        out_path: StringViewMut,
    ),
    pub write_internal_shader_cache: unsafe extern "C" fn(
        *mut BinaryIo,
        name: StringView,
        data: *const u8,
        size: usize,
        out_path: StringViewMut,
    ),
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ByteStream {
    pub data: *mut c_void,
    pub dtor: unsafe extern "C" fn(*mut ByteStream),
    pub length: unsafe extern "C" fn(*mut ByteStream) -> usize,
    pub pos: unsafe extern "C" fn(*mut ByteStream) -> usize,
    pub read: unsafe extern "C" fn(*mut ByteStream, *mut u8, usize) -> usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct PinnedMemoryOption {
    pub write_combined: bool,
}
impl Default for PinnedMemoryOption {
    fn default() -> Self {
        Self {
            write_combined: false,
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct PinnedMemoryExt {
    pub data: *mut c_void,
    pub pin_host_memory: unsafe extern "C" fn(
        *mut PinnedMemoryExt,
        *const c_void,
        usize,
        *mut c_void,
        &PinnedMemoryOption,
    ),
    pub allocate_pinned_memory: unsafe extern "C" fn(*mut PinnedMemoryExt, usize, *mut c_void),
}
pub mod denoiser_ext {
    #[repr(C)]
    #[derive(Copy, Clone, PartialEq, Eq)]
    pub enum ImageColorSpace {
        Hdr,
        LdrLinear,
        LdrSrgb,
    }
    #[repr(C)]
    #[derive(Copy, Clone, PartialEq, Eq)]
    pub enum PrefilterMode {
        None,
        Fast,
        Accurate,
    }
    #[repr(C)]
    #[derive(Copy, Clone, PartialEq, Eq)]
    pub enum FilterQuality {
        Default,
        Fast,
        Accurate,
    }
    #[repr(C)]
    #[derive(Copy, Clone, PartialEq, Eq)]
    pub enum ImageFormat {
        Float1,
        Float2,
        Float3,
        Float4,
        Half1,
        Half2,
        Half3,
        Half4,
    }
    impl ImageFormat {
        #[inline]
        pub const fn size(&self) -> usize {
            match self {
                ImageFormat::Float1 => 4,
                ImageFormat::Float2 => 8,
                ImageFormat::Float3 => 12,
                ImageFormat::Float4 => 16,
                ImageFormat::Half1 => 2,
                ImageFormat::Half2 => 4,
                ImageFormat::Half3 => 6,
                ImageFormat::Half4 => 8,
            }
        }
    }
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct Image {
        pub format: ImageFormat,
        pub buffer_handle: u64,
        pub device_ptr: *mut std::ffi::c_void,
        pub offset: usize,
        pub pixel_stride: usize,
        pub row_stride: usize,
        pub size_bytes: usize,
        pub color_space: ImageColorSpace,
        pub input_scale: f32,
    }
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct Feature {
        pub name: *const std::ffi::c_char,
        pub name_len: usize,
        pub image: Image,
    }
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct DenoiserInput {
        pub inputs: *const Image,
        pub inputs_count: usize,
        pub outputs: *const Image,
        pub features: *const Feature,
        pub features_count: usize,
        pub prefilter_mode: PrefilterMode,
        pub filter_quality: FilterQuality,
        pub noisy_features: bool,
        pub width: u32,
        pub height: u32,
    }
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct Denoiser {
        _unused: [u8; 0],
    }
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct DenoiserExt {
        pub data: *mut std::ffi::c_void,
        pub create: unsafe extern "C" fn(*const DenoiserExt, stream: u64) -> *mut Denoiser,
        pub init: unsafe extern "C" fn(*const DenoiserExt, *mut Denoiser, &DenoiserInput),
        pub execute: unsafe extern "C" fn(*const DenoiserExt, *mut Denoiser, bool),
        pub destroy: unsafe extern "C" fn(*const DenoiserExt, *mut Denoiser),
    }
    impl DenoiserExt {
        pub fn valid(&self) -> bool {
            self.data != std::ptr::null_mut()
        }
    }
}
pub use denoiser_ext::DenoiserExt;
pub fn __dummy() {}
