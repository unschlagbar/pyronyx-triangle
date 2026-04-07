#[cfg(debug_assertions)]
use std::ffi::c_void;
use std::{collections::HashSet, ffi::CStr, slice};

#[cfg(debug_assertions)]
use pyronyx::ext::{self, debug_utils::DebugUtilsInstance};
use pyronyx::{
    khr::{
        self,
        surface::{SurfaceInstance, SurfacePhysicalDevice},
        swapchain::{SwapchainDevice, SwapchainQueue},
    },
    raw_window_handle::{create_surface, get_required_extensions},
    vk::{self, SwapchainKHR},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// Max frames in flight
const MFIF: usize = 2;

#[cfg(debug_assertions)]
const VALIDATION_LAYERS: &[&CStr] = &[c"VK_LAYER_KHRONOS_validation"];

const DEVICE_EXTENSIONS: &[&CStr] = &[khr::swapchain::NAME];

struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family != u32::MAX && self.present_family != u32::MAX
    }

    fn new() -> Self {
        Self {
            graphics_family: u32::MAX,
            present_family: u32::MAX,
        }
    }
}

#[derive(Default)]
struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    fn choose_format(&self) -> vk::SurfaceFormatKHR {
        if let Some(format) = self.formats.iter().find(|f| {
            f.format == vk::Format::B8G8R8A8Unorm
                && f.color_space == vk::ColorSpaceKHR::SrgbNonlinear
        }) {
            *format
        } else {
            self.formats[0]
        }
    }

    fn choose_present_mode(&self) -> vk::PresentModeKHR {
        *self
            .present_modes
            .iter()
            .find(|&&p| p == vk::PresentModeKHR::Immediate)
            .unwrap_or(&vk::PresentModeKHR::Fifo)
    }

    fn choose_extent(&self, window: &Window) -> vk::Extent2D {
        if self.capabilities.current_extent.width != u32::MAX {
            self.capabilities.current_extent
        } else {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width,
                height: size.height,
            }
        }
    }
}

fn main() {
    let mut app = HelloTriangleApp::default();

    let event_loop = EventLoop::builder().build().unwrap();
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Default)]
struct HelloTriangleApp {
    window: Option<Window>,
    renderer: Option<Renderer>,
}

impl ApplicationHandler for HelloTriangleApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Hello Triangle")
                .with_inner_size(PhysicalSize {
                    width: WIDTH,
                    height: HEIGHT,
                });

            self.window = Some(event_loop.create_window(window_attributes).unwrap());
        }

        if self.renderer.is_none() {
            let mut renderer = Renderer::init(self.window.as_ref().unwrap());
            renderer.draw_frame(self.window.as_ref().unwrap());

            self.renderer = Some(renderer);
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _: WindowId,
        event: winit::event::WindowEvent,
    ) {
        let renderer = if let Some(renderer) = &mut self.renderer {
            renderer
        } else {
            return;
        };

        match event {
            WindowEvent::RedrawRequested => {
                if let Some(window) = &self.window
                    && window.inner_size().height != 0
                {
                    renderer.draw_frame(window);
                }
            }
            WindowEvent::Resized(_) => {
                let window = self.window.as_ref().unwrap();
                renderer.recreate_swapchain(window);
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Destroyed => event_loop.exit(),
            _ => (),
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        if let Some(renderer) = &mut self.renderer {
            renderer.cleanup();
        }
    }
}

struct Renderer {
    instance: vk::Instance,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    device: vk::Device,

    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,

    renderpass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    command_pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; MFIF],

    image_available_semaphores: [vk::Semaphore; MFIF],
    render_finish_semaphores: Vec<vk::Semaphore>,
    in_flight_fence: [vk::Fence; MFIF],
    current_frame: usize,
}

impl Renderer {
    fn init(window: &Window) -> Self {
        let instance = create_instance(window);

        #[cfg(debug_assertions)]
        let debug_messenger = setup_debug_messenger(&instance);

        let surface = {
            let display_handle = window.display_handle().unwrap().as_raw();
            let window_handle = window.window_handle().unwrap().as_raw();
            create_surface(&instance, display_handle, window_handle)
                .expect("failed to create window surface!")
        };

        let physical_device = pick_physical_device(&instance, surface);
        let (device, graphics_queue, present_queue) =
            create_logical_device(&instance, physical_device, surface);

        let mut renderer = Self {
            instance,
            #[cfg(debug_assertions)]
            debug_messenger,
            surface,

            physical_device,
            device,

            graphics_queue,
            present_queue,

            swapchain: Default::default(),
            swapchain_images: Default::default(),
            swapchain_format: Default::default(),
            swapchain_extent: Default::default(),
            swapchain_image_views: Default::default(),
            framebuffers: Default::default(),

            renderpass: Default::default(),
            pipeline_layout: Default::default(),
            pipeline: Default::default(),

            command_pool: Default::default(),
            command_buffers: Default::default(),

            image_available_semaphores: Default::default(),
            render_finish_semaphores: Default::default(),
            in_flight_fence: Default::default(),
            current_frame: 0,
        };

        renderer.swapchain = renderer.create_swapchain(window);
        renderer.get_swapchain_images();
        renderer.create_image_views();
        renderer.create_renderpass();
        renderer.create_graphics_pipeline();
        renderer.create_framebuffers();
        renderer.create_command_pool();
        renderer.create_command_buffers();
        renderer.create_sync_objects();

        renderer
    }

    fn create_swapchain(&mut self, window: &Window) -> SwapchainKHR {
        let indices = find_queue_families(&self.physical_device, self.surface);
        let swapchain_support = query_swapchain_support(&self.physical_device, self.surface);

        let surface_format = swapchain_support.choose_format();
        let present_mode = swapchain_support.choose_present_mode();
        let extent = swapchain_support.choose_extent(window);

        let image_count = swapchain_support.capabilities.min_image_count;
        let old_swapchain = self.swapchain;

        let mut create_info = vk::SwapchainCreateInfoKHR {
            surface: self.surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::ColorAttachment,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::Opaque,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain,
            ..Default::default()
        };

        let family_indices = [indices.graphics_family, indices.present_family];

        if indices.graphics_family != indices.present_family {
            create_info.image_sharing_mode = vk::SharingMode::Concurrent;
            create_info.queue_family_index_count = 2;
            create_info.queue_family_indices = family_indices.as_ptr();
        } else {
            create_info.image_sharing_mode = vk::SharingMode::Exclusive;
        }

        self.swapchain_format = surface_format.format;
        self.swapchain_extent = extent;

        self.device
            .create_swapchain(&create_info, None)
            .expect("failed to create swap chain!")
    }

    fn get_swapchain_images(&mut self) {
        self.swapchain_images = self.device.get_swapchain_images(self.swapchain).unwrap();
    }

    fn create_image_views(&mut self) {
        let mut create_info = vk::ImageViewCreateInfo {
            view_type: vk::ImageViewType::Type2d,
            format: self.swapchain_format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::Identity,
                g: vk::ComponentSwizzle::Identity,
                b: vk::ComponentSwizzle::Identity,
                a: vk::ComponentSwizzle::Identity,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::Color,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        self.swapchain_image_views = self
            .swapchain_images
            .iter()
            .map(|i| {
                create_info.image = *i;
                self.device
                    .create_image_view(&create_info, None)
                    .expect("failed to create image views!")
            })
            .collect();
    }

    fn create_renderpass(&mut self) {
        let color_attachment = vk::AttachmentDescription {
            format: self.swapchain_format,
            samples: vk::SampleCountFlags::Type1,
            load_op: vk::AttachmentLoadOp::Clear,
            store_op: vk::AttachmentStoreOp::Store,
            stencil_load_op: vk::AttachmentLoadOp::DontCare,
            stencil_store_op: vk::AttachmentStoreOp::DontCare,
            initial_layout: vk::ImageLayout::Undefined,
            final_layout: vk::ImageLayout::PresentSrcKHR,
            ..Default::default()
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::ColorAttachmentOptimal,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::Graphics,
            color_attachment_count: 1,
            color_attachments: &color_attachment_ref,
            ..Default::default()
        };

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::ColorAttachmentOutput,
            src_access_mask: vk::AccessFlags::empty(),
            dst_stage_mask: vk::PipelineStageFlags::ColorAttachmentOutput,
            dst_access_mask: vk::AccessFlags::ColorAttachmentWrite,
            dependency_flags: vk::DependencyFlags::empty(),
        };

        let create_info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            attachments: &color_attachment,
            subpass_count: 1,
            subpasses: &subpass,
            dependency_count: 1,
            dependencies: &dependency,
            ..Default::default()
        };

        self.renderpass = self
            .device
            .create_render_pass(&create_info, None)
            .expect("failed to create render pass!");
    }

    fn create_graphics_pipeline(&mut self) {
        let vert_shader_code = include_bytes!("../spv/base.vert.spv");
        let frag_shader_code = include_bytes!("../spv/base.frag.spv");

        let vert_shader_module = create_shader_module(&self.device, vert_shader_code);
        let frag_shader_module = create_shader_module(&self.device, frag_shader_code);

        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::Vertex,
            module: vert_shader_module,
            name: c"main".as_ptr(),
            ..Default::default()
        };

        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::Fragment,
            module: frag_shader_module,
            name: c"main".as_ptr(),
            ..Default::default()
        };

        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TriangleList,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::Fill,
            cull_mode: vk::CullModeFlags::Back,
            front_face: vk::FrontFace::Clockwise,
            depth_bias_enable: vk::FALSE,
            line_width: 1.0,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::Type1,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            blend_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::Copy,
            attachment_count: 1,
            attachments: &color_blend_attachment,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::Viewport, vk::DynamicState::Scissor];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let layout_create_info = vk::PipelineLayoutCreateInfo::default();

        self.pipeline_layout = self
            .device
            .create_pipeline_layout(&layout_create_info, None)
            .expect("failed to create pipeline layout!");

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            stage_count: shader_stages.len() as u32,
            stages: shader_stages.as_ptr(),
            vertex_input_state: &vertex_input_info,
            input_assembly_state: &input_assembly,
            viewport_state: &viewport_state,
            rasterization_state: &rasterizer,
            multisample_state: &multisampling,
            color_blend_state: &color_blending,
            dynamic_state: &dynamic_state,
            layout: self.pipeline_layout,
            render_pass: self.renderpass,
            subpass: 0,
            ..Default::default()
        };

        self.device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
                slice::from_mut(&mut self.pipeline),
            )
            .expect("failed to create graphics pipeline!");

        self.device.destroy_shader_module(vert_shader_module, None);
        self.device.destroy_shader_module(frag_shader_module, None);
    }

    fn create_framebuffers(&mut self) {
        let mut create_info = vk::FramebufferCreateInfo {
            render_pass: self.renderpass,
            attachment_count: 1,
            width: self.swapchain_extent.width,
            height: self.swapchain_extent.height,
            layers: 1,
            ..Default::default()
        };

        self.framebuffers = self
            .swapchain_image_views
            .iter()
            .map(|v| {
                create_info.attachments = v;
                self.device
                    .create_framebuffer(&create_info, None)
                    .expect("failed to create framebuffer!")
            })
            .collect()
    }

    fn create_command_pool(&mut self) {
        let queue_index = find_queue_families(&self.physical_device, self.surface).graphics_family;

        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::ResetCommandBuffer,
            queue_family_index: queue_index,
            ..Default::default()
        };

        self.command_pool = self
            .device
            .create_command_pool(&create_info, None)
            .expect("failed to create command pool!");
    }

    fn create_command_buffers(&mut self) {
        let allocate_info = vk::CommandBufferAllocateInfo {
            command_pool: self.command_pool,
            level: vk::CommandBufferLevel::Primary,
            command_buffer_count: MFIF as u32,
            ..Default::default()
        };

        unsafe {
            self.device
                .allocate_command_buffers(&allocate_info, &mut self.command_buffers)
                .expect("failed to allocate command buffers!")
        };
    }

    fn create_sync_objects(&mut self) {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::Signaled,
            ..Default::default()
        };

        self.render_finish_semaphores
            .resize(self.swapchain_images.len(), vk::Semaphore::null());

        self.image_available_semaphores.iter_mut().for_each(|s| {
            *s = self
                .device
                .create_semaphore(&semaphore_create_info, None)
                .expect("failed to create synchronization objects")
        });

        self.render_finish_semaphores.iter_mut().for_each(|s| {
            *s = self
                .device
                .create_semaphore(&semaphore_create_info, None)
                .expect("failed to create synchronization objects")
        });
        self.in_flight_fence.iter_mut().for_each(|s| {
            *s = self
                .device
                .create_fence(&fence_create_info, None)
                .expect("failed to create synchronization objects")
        });
    }

    fn draw_frame(&mut self, window: &Window) {
        let current = self.current_frame;

        self.device
            .wait_for_fences(&[self.in_flight_fence[current]], true, u64::MAX)
            .unwrap();

        let result = self.device.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available_semaphores[current],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok(index) => index,
            Err(vk::Error::OutOfDateKHR | vk::Error::SuboptimalKHR) => {
                self.recreate_swapchain(window);
                return;
            }
            _ => panic!("failed to acquire swap chain image!"),
        };

        self.device
            .reset_fences(&[self.in_flight_fence[current]])
            .unwrap();

        self.command_buffers[current]
            .reset(vk::CommandBufferResetFlags::empty())
            .unwrap();

        self.record_command_buffer(image_index);

        let wait_semaphores = [self.image_available_semaphores[current]];
        let wait_stages = [vk::PipelineStageFlags::ColorAttachmentOutput];
        let signal_semaphores = [self.render_finish_semaphores[image_index as usize]];

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: wait_semaphores.len() as u32,
            wait_semaphores: wait_semaphores.as_ptr(),
            wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            command_buffers: &self.command_buffers[current].handle(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        self.graphics_queue
            .submit(&[submit_info], self.in_flight_fence[current])
            .expect("failed to submit draw command buffer!");

        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: signal_semaphores.len() as u32,
            wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            swapchains: &self.swapchain,
            image_indices: &image_index,
            ..Default::default()
        };

        match self.present_queue.present(&present_info) {
            Ok(_) => (),
            Err(vk::Error::OutOfDateKHR | vk::Error::SuboptimalKHR) => {
                println!("Recreate Swapchain");
            }
            Err(_) => panic!("failed to present swap chain image!"),
        }

        self.current_frame = (current + 1) % MFIF;
    }

    fn record_command_buffer(&mut self, image_index: u32) {
        let cmd_buf = self.command_buffers[self.current_frame];
        let begin_info = vk::CommandBufferBeginInfo::default();

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        };

        let render_pass_begin = vk::RenderPassBeginInfo {
            render_pass: self.renderpass,
            framebuffer: self.framebuffers[image_index as usize],
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swapchain_extent,
            },
            clear_value_count: 1,
            clear_values: &clear_color,
            ..Default::default()
        };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swapchain_extent.width as f32,
            height: self.swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: self.swapchain_extent,
        };

        cmd_buf
            .begin(&begin_info)
            .expect("failed to begin recording command buffer!");

        cmd_buf.begin_render_pass(&render_pass_begin, vk::SubpassContents::Inline);
        cmd_buf.bind_pipeline(vk::PipelineBindPoint::Graphics, self.pipeline);

        cmd_buf.set_viewport(0, &[viewport]);
        cmd_buf.set_scissor(0, &[scissor]);

        cmd_buf.draw(3, 1, 0, 0);

        cmd_buf.end_render_pass();
        cmd_buf.end().expect("failed to record command buffer!");
    }

    fn recreate_swapchain(&mut self, window: &Window) {
        let size = window.inner_size();

        if size.height == 0 || size.width == 0 {
            return;
        }

        self.device.wait_idle().unwrap();

        let swapchain = self.create_swapchain(window);
        self.cleanup_swapchain();
        self.swapchain = swapchain;

        self.get_swapchain_images();
        self.create_image_views();
        self.create_framebuffers();
    }

    fn cleanup_swapchain(&mut self) {
        for i in 0..self.framebuffers.len() {
            self.device.destroy_framebuffer(self.framebuffers[i], None);
            self.device
                .destroy_image_view(self.swapchain_image_views[i], None);
        }

        self.device.destroy_swapchain(self.swapchain, None);
    }

    fn cleanup(&mut self) {
        self.device.wait_idle().unwrap();

        self.cleanup_swapchain();

        self.device.destroy_pipeline(self.pipeline, None);
        self.device
            .destroy_pipeline_layout(self.pipeline_layout, None);

        self.device.destroy_render_pass(self.renderpass, None);

        for i in 0..MFIF {
            self.device
                .destroy_semaphore(self.image_available_semaphores[i], None);
            self.device.destroy_fence(self.in_flight_fence[i], None);
        }

        for &semaphore in &self.render_finish_semaphores {
            self.device.destroy_semaphore(semaphore, None);
        }

        self.device.destroy_command_pool(self.command_pool, None);
        self.device.destroy(None);

        #[cfg(debug_assertions)]
        self.instance
            .destroy_debug_utils_messenger(self.debug_messenger, None);

        self.instance.destroy_surface(self.surface, None);
        self.instance.destroy(None);
    }
}

fn create_instance(window: &Window) -> vk::Instance {
    #[cfg(debug_assertions)]
    let validation = if check_validation_layer_support() {
        true
    } else {
        println!("validation layers requested, but not available!");
        false
    };

    let app_info = vk::ApplicationInfo {
        application_name: c"Hello Triangle".as_ptr(),
        application_version: vk::make_api_version(0, 1, 0, 0),
        engine_name: c"No Engine".as_ptr(),
        engine_version: vk::make_api_version(0, 1, 0, 0),
        api_version: vk::API_VERSION_1_0,
        ..Default::default()
    };

    let display_handle = window.display_handle().unwrap().as_raw();
    let extensions = get_required_extensions(display_handle).unwrap();

    #[cfg(debug_assertions)]
    let extensions = [
        extensions[0],
        extensions[1],
        ext::debug_utils::NAME.as_ptr(),
    ];

    let layers: &[&CStr] = if cfg!(debug_assertions) && validation {
        VALIDATION_LAYERS
    } else {
        &[]
    };

    let create_info = vk::InstanceCreateInfo {
        application_info: &app_info,
        enabled_extension_count: extensions.len() as u32,
        enabled_extension_names: extensions.as_ptr(),
        enabled_layer_count: layers.len() as u32,
        enabled_layer_names: layers.as_ptr().cast(),
        ..Default::default()
    };

    #[cfg(debug_assertions)]
    let mut debug_create_info = populate_debug_messenger_create_info();

    #[cfg(debug_assertions)]
    let create_info = create_info.next(&mut debug_create_info);

    unsafe { vk::Instance::create(&create_info, None).expect("failed to create instance!") }
}

#[cfg(debug_assertions)]
fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT {
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::Error
            | vk::DebugUtilsMessageSeverityFlagsEXT::Warning
            | vk::DebugUtilsMessageSeverityFlagsEXT::Verbose,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::General
            | vk::DebugUtilsMessageTypeFlagsEXT::Performance
            | vk::DebugUtilsMessageTypeFlagsEXT::Validation,
        pfn_user_callback: Some(debug_callback),
        ..Default::default()
    }
}

#[cfg(debug_assertions)]
fn setup_debug_messenger(instance: &vk::Instance) -> vk::DebugUtilsMessengerEXT {
    #[cfg(debug_assertions)]
    let create_info = populate_debug_messenger_create_info();
    instance
        .create_debug_utils_messenger(&create_info, None)
        .expect("failed to set up debug messenger!")
}

fn pick_physical_device(instance: &vk::Instance, surface: vk::SurfaceKHR) -> vk::PhysicalDevice {
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("failed to find GPUs with Vulkan support!")
    };

    devices
        .into_iter()
        .find(|d| is_device_suitable(d, surface))
        .expect("failed to find a suitable GPU!")
}

fn create_logical_device(
    instance: &vk::Instance,
    device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> (vk::Device, vk::Queue, vk::Queue) {
    let indices = find_queue_families(&device, surface);
    let graphics_family = indices.graphics_family;
    let present_family = indices.present_family;

    let queue_priority = 1.0;

    let queue_create_infos: &[_] = if graphics_family == present_family {
        &[vk::DeviceQueueCreateInfo {
            queue_family_index: graphics_family,
            queue_priorities: &queue_priority,
            queue_count: 1,
            ..Default::default()
        }]
    } else {
        &[
            vk::DeviceQueueCreateInfo {
                queue_family_index: graphics_family,
                queue_priorities: &queue_priority,
                queue_count: 1,
                ..Default::default()
            },
            vk::DeviceQueueCreateInfo {
                queue_family_index: present_family,
                queue_priorities: &queue_priority,
                queue_count: 1,
                ..Default::default()
            },
        ]
    };

    let layers: &[&CStr] = if cfg!(debug_assertions) && check_validation_layer_support() {
        VALIDATION_LAYERS
    } else {
        &[]
    };

    let create_info = vk::DeviceCreateInfo {
        queue_create_info_count: queue_create_infos.len() as u32,
        queue_create_infos: queue_create_infos.as_ptr(),
        enabled_layer_count: layers.len() as u32,
        enabled_layer_names: layers.as_ptr().cast(),
        enabled_extension_count: DEVICE_EXTENSIONS.len() as u32,
        enabled_extension_names: DEVICE_EXTENSIONS.as_ptr().cast(),
        enabled_features: &vk::PhysicalDeviceFeatures::default(),
        ..Default::default()
    };

    let logical_device = unsafe {
        device
            .create_device(&create_info, None, instance)
            .expect("failed to create logical device!")
    };

    let graphics_queue = unsafe { logical_device.get_device_queue(graphics_family, 0) };
    let present_queue = unsafe { logical_device.get_device_queue(present_family, 0) };

    (logical_device, graphics_queue, present_queue)
}

fn create_shader_module(device: &vk::Device, code: &[u8]) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo {
        code_size: code.len(),
        code: code.as_ptr().cast(),
        ..Default::default()
    };

    device
        .create_shader_module(&create_info, None)
        .expect("failed to create shader module!")
}

fn query_swapchain_support(
    device: &vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> SwapchainSupportDetails {
    SwapchainSupportDetails {
        capabilities: device.get_surface_capabilities(surface).unwrap(),
        formats: device.get_surface_formats(surface).unwrap(),
        present_modes: device.get_surface_present_modes(surface).unwrap(),
    }
}

fn is_device_suitable(device: &vk::PhysicalDevice, surface: vk::SurfaceKHR) -> bool {
    let indices = find_queue_families(device, surface);
    let extensions_supported = check_device_extension_support(device);

    let swap_chain_adequate = if extensions_supported {
        let swap_chain_support = query_swapchain_support(device, surface);
        !swap_chain_support.formats.is_empty() && !swap_chain_support.present_modes.is_empty()
    } else {
        false
    };
    indices.is_complete() && swap_chain_adequate
}

fn check_device_extension_support(device: &vk::PhysicalDevice) -> bool {
    let available_extensions = device.enumerate_device_extension_properties(None).unwrap();

    let mut required_extension: HashSet<&CStr> =
        HashSet::from_iter(DEVICE_EXTENSIONS.iter().copied());

    for availabe in &available_extensions {
        let available_name = unsafe { CStr::from_ptr(availabe.extension_name.as_ptr()) };
        required_extension.remove(available_name);
    }

    required_extension.is_empty()
}

fn find_queue_families(device: &vk::PhysicalDevice, surface: vk::SurfaceKHR) -> QueueFamilyIndices {
    let mut indices = QueueFamilyIndices::new();

    let queue_families = device.get_queue_family_properties();

    for (i, queue_family) in queue_families.iter().enumerate() {
        if queue_family.queue_flags.contains(vk::QueueFlags::Graphics) {
            indices.graphics_family = i as u32;
        }

        if device
            .get_surface_support(i as u32, surface)
            .unwrap_or_default()
        {
            indices.present_family = i as u32;
        }

        if indices.is_complete() {
            break;
        }
    }

    indices
}

#[cfg(debug_assertions)]
fn check_validation_layer_support() -> bool {
    let available_layers = vk::enumerate_instance_layer_properties().unwrap();

    for &layer_name in VALIDATION_LAYERS {
        let mut layer_found = false;

        'a: for layer_props in &available_layers {
            let available_name = unsafe { CStr::from_ptr(layer_props.layer_name.as_ptr()) };
            if layer_name == available_name {
                layer_found = true;
                break 'a;
            }
        }

        if !layer_found {
            return false;
        }
    }

    true
}

#[cfg(debug_assertions)]
extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> u32 {
    let message = unsafe { CStr::from_ptr((*callback_data).message) };
    println!(
        "[Debug][{}][{}] {:?}",
        message_severity, message_type, message
    );
    0
}
