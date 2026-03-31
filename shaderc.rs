use std::fs;
use std::io::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

const SHADER_DIR: &str = "shaders";
const SPV_DIR: &str = "spv";

#[cfg(target_os = "windows")]
const GLSLC: &str = "/Bin/glslc.exe";

#[cfg(target_os = "windows")]
const SLANGC: &str = "/Bin/slangc.exe";

#[cfg(not(target_os = "windows"))]
const GLSLC: &str = "/bin/glslc";

#[cfg(not(target_os = "windows"))]
const SLANGC: &str = "/bin/slangc";

pub fn build() {
    println!("cargo:rerun-if-changed={SHADER_DIR}");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let compilers = match option_env!("VULKAN_SDK") {
        Some(vulkan_sdk) => (
            vulkan_sdk.to_string() + GLSLC,
            vulkan_sdk.to_string() + SLANGC,
        ),
        None => {
            println!("cargo::warning=Vulkan SDK env variable not set");
            return;
        }
    };

    let _ = fs::create_dir(SPV_DIR);

    for shader_path in get_shader_files(SHADER_DIR).unwrap() {
        compile_shader(&shader_path, &compilers);
    }
}

fn get_shader_files(dir: &str) -> Result<Vec<PathBuf>, Error> {
    let mut shader_files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let path = entry?.path();

        if path.is_dir() {
            shader_files.extend(get_shader_files(path.to_str().unwrap())?);
        } else if let Some(extension) = path.extension()
            && (extension == "frag"
                || extension == "vert"
                || extension == "comp"
                || extension == "geom"
                || extension == "slang")
        {
            shader_files.push(path);
        }
    }
    Ok(shader_files)
}

fn get_spirv_output_path(shader_path: &Path) -> PathBuf {
    let name = shader_path.file_name().unwrap().to_str().unwrap();
    let name = name.strip_suffix(".slang").unwrap_or(name);
    PathBuf::from(format!("{SPV_DIR}/{name}.spv"))
}

fn compile_shader(shader_path: &Path, compilers: &(String, String)) {
    let output_path = get_spirv_output_path(shader_path);

    let output = if shader_path.extension().is_some_and(|ext| ext == "slang") {
        Command::new(&compilers.1)
            .arg(shader_path)
            .arg("-target")
            .arg("spirv")
            .arg("-o")
            .arg(&output_path)
            .output()
    } else {
        Command::new(&compilers.0)
            .arg(shader_path)
            .arg("-o")
            .arg(&output_path)
            .output()
    };

    match output {
        Ok(out) => {
            if !out.status.success() {
                println!(
                    "cargo::error=Shader compilation failed for: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            } else {
                println!("cargo:info=Compiled Shader {shader_path:?}");
            }
        }
        Err(e) => println!("cargo::warning=Shader build command error: {e}"),
    }
}
