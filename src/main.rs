use std::collections::HashSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, ErrorKind};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use anyhow::{bail, Context};
use nix::fcntl::{Flock, FlockArg};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use glob::glob;

const IN_NIX_STORE: bool = false;
const CACHE_VERSION: i32 = 3;
const PATCHELF_PATH: &str = if IN_NIX_STORE {
    "@patchelf-bin@"
} else {
    "patchelf"
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResolvedLib {
    name: String,
    dirpath: String,
    fullpath: String,
    last_modification: f64,
    size: u64,
}

impl ResolvedLib {
    fn new(name: String, dirpath: String, fullpath: String) -> anyhow::Result<Self> {
        let metadata = fs::metadata(&fullpath)?;
        let last_modification = metadata
            .modified()
            .with_context(|| format!("failed to get metadata for resolved lib {fullpath}"))?
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let size = metadata.len();

        Ok(ResolvedLib {
            name,
            dirpath,
            fullpath,
            last_modification,
            size,
        })
    }
}

impl PartialEq for ResolvedLib {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.fullpath == other.fullpath
            && self.dirpath == other.dirpath
            && self.last_modification == other.last_modification
            && self.size == other.size
    }
}

impl Eq for ResolvedLib {}

impl Hash for ResolvedLib {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.dirpath.hash(state);
        self.fullpath.hash(state);
        self.last_modification.to_bits().hash(state);
        self.size.hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LibraryPath {
    glx: Vec<ResolvedLib>,
    cuda: Vec<ResolvedLib>,
    generic: Vec<ResolvedLib>,
    egl: Vec<ResolvedLib>,
    path: String,
}

impl PartialEq for LibraryPath {
    fn eq(&self, other: &Self) -> bool {
        let glx_set: HashSet<_> = self.glx.iter().collect();
        let other_glx_set: HashSet<_> = other.glx.iter().collect();
        let cuda_set: HashSet<_> = self.cuda.iter().collect();
        let other_cuda_set: HashSet<_> = other.cuda.iter().collect();
        let generic_set: HashSet<_> = self.generic.iter().collect();
        let other_generic_set: HashSet<_> = other.generic.iter().collect();
        let egl_set: HashSet<_> = self.egl.iter().collect();
        let other_egl_set: HashSet<_> = other.egl.iter().collect();

        glx_set == other_glx_set
            && cuda_set == other_cuda_set
            && generic_set == other_generic_set
            && egl_set == other_egl_set
            && self.path == other.path
    }
}

impl Eq for LibraryPath {}

impl Hash for LibraryPath {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for lib in &self.glx {
            lib.hash(state);
        }
        for lib in &self.cuda {
            lib.hash(state);
        }
        for lib in &self.generic {
            lib.hash(state);
        }
        for lib in &self.egl {
            lib.hash(state);
        }
        self.path.hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheDirContent {
    paths: Vec<LibraryPath>,
    version: i32,
}

impl CacheDirContent {
    fn new(paths: Vec<LibraryPath>) -> Self {
        CacheDirContent {
            paths,
            version: CACHE_VERSION,
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl PartialEq for CacheDirContent {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version && {
            let self_paths: HashSet<_> = self.paths.iter().collect();
            let other_paths: HashSet<_> = other.paths.iter().collect();
            self_paths == other_paths
        }
    }
}
lazy_static::lazy_static! {
    static ref NVIDIA_DSO_PATTERNS: Vec<Regex> = vec![
        Regex::new(r"libGLESv1_CM_nvidia\.so.*$").unwrap(),
        Regex::new(r"libGLESv2_nvidia\.so.*$").unwrap(),
        Regex::new(r"libglxserver_nvidia\.so.*$").unwrap(),
        Regex::new(r"libnvcuvid\.so.*$").unwrap(),
        Regex::new(r"libnvidia-allocator\.so.*$").unwrap(),
        Regex::new(r"libnvidia-cfg\.so.*$").unwrap(),
        Regex::new(r"libnvidia-compiler\.so.*$").unwrap(),
        Regex::new(r"libnvidia-eglcore\.so.*$").unwrap(),
        Regex::new(r"libnvidia-encode\.so.*$").unwrap(),
        Regex::new(r"libnvidia-fbc\.so.*$").unwrap(),
        Regex::new(r"libnvidia-glcore\.so.*$").unwrap(),
        Regex::new(r"libnvidia-glsi\.so.*$").unwrap(),
        Regex::new(r"libnvidia-glvkspirv\.so.*$").unwrap(),
        Regex::new(r"libnvidia-gpucomp\.so.*$").unwrap(),
        Regex::new(r"libnvidia-ngx\.so.*$").unwrap(),
        Regex::new(r"libnvidia-nvvm\.so.*$").unwrap(),
        Regex::new(r"libnvidia-opencl\.so.*$").unwrap(),
        Regex::new(r"libnvidia-opticalflow\.so.*$").unwrap(),
        Regex::new(r"libnvidia-ptxjitcompiler\.so.*$").unwrap(),
        Regex::new(r"libnvidia-rtcore\.so.*$").unwrap(),
        Regex::new(r"libnvidia-tls\.so.*$").unwrap(),
        Regex::new(r"libnvidia-vulkan-producer\.so.*$").unwrap(),
        Regex::new(r"libnvidia-wayland-client\.so.*$").unwrap(),
        Regex::new(r"libnvoptix\.so.*$").unwrap(),
        Regex::new(r"libnvtegrahv\.so.*$").unwrap(),
        Regex::new(r"libdrm\.so.*$").unwrap(),
        Regex::new(r"libffi\.so.*$").unwrap(),
        Regex::new(r"libgbm\.so.*$").unwrap(),
        Regex::new(r"libexpat\.so.*$").unwrap(),
        Regex::new(r"libxcb-glx\.so.*$").unwrap(),
        Regex::new(r"libX11-xcb\.so.*$").unwrap(),
        Regex::new(r"libX11\.so.*$").unwrap(),
        Regex::new(r"libXext\.so.*$").unwrap(),
        Regex::new(r"libwayland-server\.so.*$").unwrap(),
        Regex::new(r"libwayland-client\.so.*$").unwrap(),
        Regex::new(r"libd3d12core\.so.*$").unwrap(),
        Regex::new(r"libd3d12\.so.*$").unwrap(),
        Regex::new(r"libdxcore\.so.*$").unwrap(),
    ];

    static ref CUDA_DSO_PATTERNS: Vec<Regex> = vec![
        Regex::new(r"libcudadebugger\.so.*$").unwrap(),
        Regex::new(r"libcuda\.so.*$").unwrap(),
        Regex::new(r"libnvidia-ml\.so.*$").unwrap(),
    ];

    static ref GLX_DSO_PATTERNS: Vec<Regex> = vec![
        Regex::new(r"libGLX_nvidia\.so.*$").unwrap(),
    ];

    static ref EGL_DSO_PATTERNS: Vec<Regex> = vec![
        Regex::new(r"libEGL_nvidia\.so.*$").unwrap(),
        Regex::new(r"libnvidia-egl-wayland\.so.*$").unwrap(),
        Regex::new(r"libnvidia-egl-gbm\.so.*$").unwrap(),
    ];
}

fn parse_ld_conf_file(ld_conf_file_path: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let file = fs::File::open(ld_conf_file_path).unwrap();
    for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("#") {
            continue;
        }
        if line.starts_with("include ") {
            let mut dirglob = line.trim_start_matches("include ").to_string();
            if !dirglob.starts_with("/") {
                let mut search_path = ld_conf_file_path.canonicalize().unwrap().parent().unwrap().to_str().unwrap().to_string();
                search_path += "/";
                search_path += &dirglob;
                dirglob = search_path;
            }
            for entry in glob(&dirglob).unwrap().flatten() {
                paths.extend(parse_ld_conf_file(entry.as_path()));
            }
            continue;
        }
        paths.push(Path::new(line).to_path_buf());
    }
    paths
}

fn get_ld_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // Add LD_LIBRARY_PATH paths
    if let Ok(ld_path) = env::var("LD_LIBRARY_PATH") {
        paths.extend(ld_path.split(':').map(PathBuf::from));
    }

    // Add paths from ld.so.conf file
    let ld_conf_file_path = Path::new("/etc/ld.so.conf");
    if ld_conf_file_path.exists() {
        paths.extend(parse_ld_conf_file(ld_conf_file_path));
    }

    // Add PREFIX paths if available
    if let Ok(prefix) = env::var("PREFIX") {
        let prefix_path = PathBuf::from(prefix);
        paths.extend([
            prefix_path.join("lib"),
            prefix_path.join("usr/lib"),
            prefix_path.join("lib64"),
            prefix_path.join("usr/lib64"),
        ]);
    }

    // Add standard paths
    paths.extend([
        PathBuf::from("/lib"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/lib64"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/run/opengl-driver/lib"),
        PathBuf::from("/usr/lib/wsl/lib"),
    ]);

    // Filter only existing directories
    paths.into_iter().filter(|p| p.is_dir()).collect()
}

fn resolve_libraries(path: &Path, patterns: &[Regex]) -> Vec<ResolvedLib> {
    let mut libraries = Vec::new();

    let is_dso_matching_pattern =
        |filename: &str| patterns.iter().any(|pattern| pattern.is_match(filename));

    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.filter_map(Result::ok) {
            let file_path = entry.path();
            if file_path.is_file() {
                if let Some(file_name) = file_path.file_name().and_then(|n| n.to_str()) {
                    if is_dso_matching_pattern(file_name) {
                        if let Ok(lib) = ResolvedLib::new(
                            file_name.to_string(),
                            path.to_string_lossy().into_owned(),
                            file_path.to_string_lossy().into_owned(),
                        ) {
                            libraries.push(lib);
                        }
                    }
                }
            }
        }
    }

    libraries
}

fn copy_and_patch_libs(
    dsos: &[ResolvedLib],
    dest_dir: &Path,
    rpath: Option<&Path>,
) -> anyhow::Result<()> {
    let rpath = rpath.unwrap_or(dest_dir);

    for dso in dsos {
        let basename = Path::new(&dso.fullpath).file_name().unwrap();
        let newpath = dest_dir.join(basename);

        log_info(&format!("Copying and patching {:?} to {:?}", dso, newpath));

        fs::copy(&dso.fullpath, &newpath)?;

        let mut perms = fs::metadata(&newpath)?.permissions();
        perms.set_mode(perms.mode() | 0o200); // Add write permission
        fs::set_permissions(&newpath, perms)?;
    }

    let new_paths: Vec<_> = dsos
        .iter()
        .map(|dso| dest_dir.join(Path::new(&dso.fullpath).file_name().unwrap()))
        .collect();

    patch_dsos(&new_paths, rpath)?;
    Ok(())
}

fn log_info(message: &str) {
    if env::var("DEBUG").is_ok() {
        eprintln!("[+] {}", message);
    }
}

fn patch_dsos(dso_paths: &[PathBuf], rpath: &Path) -> std::io::Result<()> {
    log_info(&format!("Patching {:?}", dso_paths));

    let mut command = Command::new(PATCHELF_PATH);
    command
        .arg("--set-rpath")
        .arg(rpath)
        .args(dso_paths.iter().map(|p| p.as_os_str()));

    let output = command.output()?;

    if !output.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Patchelf failed with status: {}", output.status),
        ));
    }

    Ok(())
}
#[derive(Serialize, Deserialize)]
struct EglConfig {
    file_format_version: String,
    icd: EglIcdConfig,
}

#[derive(Serialize, Deserialize)]
struct EglIcdConfig {
    library_path: String,
}

fn generate_nvidia_egl_config_files(egl_conf_dir: &Path) -> anyhow::Result<()> {
    let dso_paths = vec![
        ("10_nvidia.json", "libEGL_nvidia.so.0"),
        ("10_nvidia_wayland.json", "libnvidia-egl-wayland.so.1"),
        ("15_nvidia_gbm.json", "libnvidia-egl-gbm.so.1"),
    ];

    fs::create_dir_all(egl_conf_dir)?;

    for (conf_file_name, dso_name) in dso_paths {
        let config = EglConfig {
            file_format_version: "1.0.0".to_string(),
            icd: EglIcdConfig {
                library_path: dso_name.to_string(),
            },
        };

        let conf_path = egl_conf_dir.join(conf_file_name);
        log_info(&format!("Writing {} conf to {:?}", dso_name, egl_conf_dir));

        let json = serde_json::to_string_pretty(&config)?;
        fs::write(conf_path, json)?;
    }

    Ok(())
}

fn is_dso_cache_up_to_date(dsos: &CacheDirContent, cache_file_path: &Path) -> bool {
    log_info("Checking if the cache is up to date");

    if cache_file_path.is_file() {
        match fs::read_to_string(cache_file_path) {
            Ok(content) => match CacheDirContent::from_json(&content) {
                Ok(cached_dsos) => return dsos == &cached_dsos,
                Err(_) => return false,
            },
            Err(_) => return false,
        }
    }
    false
}

fn scan_dsos_from_dir(path: &Path) -> Option<LibraryPath> {
    let generic = resolve_libraries(path, &NVIDIA_DSO_PATTERNS);

    if !generic.is_empty() {
        let cuda = resolve_libraries(path, &CUDA_DSO_PATTERNS);
        let glx = resolve_libraries(path, &GLX_DSO_PATTERNS);
        let egl = resolve_libraries(path, &EGL_DSO_PATTERNS);

        Some(LibraryPath {
            glx,
            cuda,
            generic,
            egl,
            path: path.to_string_lossy().into_owned(),
        })
    } else {
        None
    }
}

fn cache_library_path(
    library_path: &LibraryPath,
    temp_cache_dir_root: &Path,
    final_cache_dir_root: &Path,
) -> anyhow::Result<String> {
    // Hash computation
    let mut hasher = Sha256::new();
    hasher.update(library_path.path.as_bytes());
    let path_hash = format!("{:x}", hasher.finalize());

    // Paths
    let cache_path_root = temp_cache_dir_root.join(&path_hash);
    let lib_dir = cache_path_root.join("lib");
    let rpath_lib_dir = final_cache_dir_root.join(&path_hash).join("lib");
    let cuda_dir = cache_path_root.join("cuda");
    let egl_dir = cache_path_root.join("egl");
    let glx_dir = cache_path_root.join("glx");

    // Create directories and copy/patch DSOs
    let dirs_and_dsos = [
        (&library_path.generic, &lib_dir),
        (&library_path.cuda, &cuda_dir),
        (&library_path.egl, &egl_dir),
        (&library_path.glx, &glx_dir),
    ];

    for (dsos, dir) in dirs_and_dsos.iter() {
        fs::create_dir_all(dir)?;
        if !dsos.is_empty() {
            copy_and_patch_libs(dsos, dir, Some(&rpath_lib_dir))?;
        } else {
            log_info(&format!(
                "Did not find any DSO to put in {:?}, skipping copy and patching.",
                dir
            ));
        }
    }

    Ok(path_hash)
}

fn generate_cache_ld_library_path(cache_paths: &[String]) -> String {
    let mut ld_library_paths = Vec::new();

    for path in cache_paths {
        ld_library_paths.extend([
            format!("{}/glx", path),
            format!("{}/cuda", path),
            format!("{}/egl", path),
        ]);
    }

    ld_library_paths.join(":")
}

fn generate_cache_metadata(
    cache_dir: &Path,
    cache_content: &CacheDirContent,
    cache_paths: &[String],
) -> anyhow::Result<String> {
    let cache_file_path = cache_dir.join("cache.json");
    let cached_ld_library_path = cache_dir.join("ld_library_path");
    let egl_conf_dir = cache_dir.join("egl-confs");

    // Write cache.json
    fs::write(&cache_file_path, cache_content.to_json())?;

    // Generate and write LD_LIBRARY_PATH
    let nix_gl_ld_library_path = generate_cache_ld_library_path(cache_paths);
    log_info(&format!(
        "Caching LD_LIBRARY_PATH: {}",
        nix_gl_ld_library_path
    ));
    fs::write(&cached_ld_library_path, &nix_gl_ld_library_path)?;

    // Generate EGL config files
    generate_nvidia_egl_config_files(&egl_conf_dir)?;

    Ok(nix_gl_ld_library_path)
}

use clap::Parser;
use std::collections::HashMap;

#[derive(Parser, Debug)]
struct Args {
    /// Use the driver libraries contained in this directory instead of discovering them from the load path
    #[arg(short, long)]
    driver_directory: Option<PathBuf>,

    /// Print the GL/Cuda LD_LIBRARY_PATH env you should add to your environment
    #[arg(short, long)]
    print_ld_library_path: bool,

    /// Nix-built binary you'd like to wrap
    nix_binary: Option<PathBuf>,

    /// The args passed to the wrapped binary
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

fn nvidia_main(
    cache_dir: &Path,
    dso_vendor_paths: &[PathBuf],
    print_ld_library_path: bool,
) -> anyhow::Result<HashMap<String, String>> {
    log_info("Nvidia routine begins");

    // Find Host DSOS
    log_info("Searching for the host DSOs");
    let mut cache_content = CacheDirContent::new(Vec::new());
    let cache_file_path = cache_dir.join("cache.json");
    let lock_path = cache_dir.parent().unwrap().join("nix-gl-host.lock");
    let cached_ld_library_path = cache_dir.join("ld_library_path");
    let egl_conf_dir = cache_dir.join("egl-confs");

    // Cache/Patch DSOs with file locking
    let lock_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&lock_path)
        .with_context(|| format!("failed to open DSO lock path {lock_path:?}"))?;

    log_info("Acquiring the cache lock");
    let lock = Flock::lock(lock_file, FlockArg::LockExclusive)
        .map_err(|e| std::io::Error::new(ErrorKind::AlreadyExists, e.1))
        .context("failed to acquire the cache lock")?;
    log_info("Cache lock acquired");

    for path in dso_vendor_paths {
        if let Some(res) = scan_dsos_from_dir(path) {
            cache_content.paths.push(res);
        }
    }

    let nix_gl_ld_library_path = if !is_dso_cache_up_to_date(&cache_content, &cache_file_path)
        || !cached_ld_library_path.exists()
    {
        log_info("The cache is not up to date, regenerating it");

        let tmp_dir = TempDir::new()?;
        let tmp_cache_dir = tmp_dir.path().join("nix-gl-host");
        fs::create_dir(&tmp_cache_dir)?;

        let mut cache_paths = Vec::new();
        for p in &cache_content.paths {
            log_info(&format!("Caching {:?}", p));
            cache_paths.push(cache_library_path(p, &tmp_cache_dir, cache_dir)?);
        }

        let cache_absolute_paths: Vec<_> = cache_paths
            .iter()
            .map(|p| cache_dir.join(p).to_string_lossy().into_owned())
            .collect();

        let nix_gl_ld_library_path = Some(generate_cache_metadata(
            &tmp_cache_dir,
            &cache_content,
            &cache_absolute_paths,
        )?);

        log_info(&format!("Moving {:?} to {:?}", tmp_cache_dir, cache_dir));
        if cache_dir.exists() {
            fs::remove_dir_all(cache_dir)
                .with_context(|| format!("failed to remove the old cache dir {cache_dir:?}"))?;
        }
        let renamed = fs::rename(&tmp_cache_dir, cache_dir);
        if renamed.is_err() {
            log_info("failed to atomically move cache dir because of {err}. trying backup.");
            copy_dir_all(&tmp_cache_dir, cache_dir).with_context(||
                format!("failed to move the tmp cache dir {tmp_cache_dir:?} to {cache_dir:?} using the backup strategy.")
            )?;
            fs::remove_dir_all(&tmp_cache_dir)
                .with_context(|| format!("failed to remove the tmp cache dir {tmp_cache_dir:?}"))?;
        }
        nix_gl_ld_library_path
    } else {
        log_info("The cache is up to date, re-using it.");
        Some(fs::read_to_string(&cached_ld_library_path)?)
    };

    drop(lock);
    log_info("Cache lock released");

    let nix_gl_ld_library_path =
        nix_gl_ld_library_path.expect("The nix-host-gl LD_LIBRARY_PATH is not set");

    log_info(&format!(
        "Injecting LD_LIBRARY_PATH: {}",
        nix_gl_ld_library_path
    ));

    let mut new_env = HashMap::new();
    new_env.insert(
        "__GLX_VENDOR_LIBRARY_NAME".to_string(),
        "nvidia".to_string(),
    );
    new_env.insert(
        "__EGL_VENDOR_LIBRARY_DIRS".to_string(),
        egl_conf_dir.to_string_lossy().into_owned(),
    );

    let ld_library_path = match env::var("LD_LIBRARY_PATH") {
        Ok(current) => format!("{}:{}", nix_gl_ld_library_path, current),
        Err(_) => nix_gl_ld_library_path.clone(),
    };

    if print_ld_library_path {
        println!("{}", nix_gl_ld_library_path);
    }

    new_env.insert("LD_LIBRARY_PATH".to_string(), ld_library_path);
    Ok(new_env)
}

fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> std::io::Result<()> {
    fs::create_dir_all(&dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
        }
    }
    Ok(())
}

fn exec_binary(bin_path: &Path, args: &[String]) -> std::io::Result<std::process::Child> {
    log_info(&format!("Execv-ing {:?}", bin_path));
    log_info("Goodbye now.");

    Err(Command::new(bin_path).args(args).exec())
}

fn main() -> anyhow::Result<()> {
    let opt = Args::parse();

    let start_time = SystemTime::now();
    let home = env::var("HOME").expect("HOME environment variable not set");
    let xdg_cache_home = env::var("XDG_CACHE_HOME").unwrap_or_else(|_| format!("{}/.cache", home));
    let cache_dir = PathBuf::from(xdg_cache_home).join("nix-gl-host");
    fs::create_dir_all(&cache_dir).context("failed to create cache dir")?;

    log_info(&format!("Using {:?} as cache dir.", cache_dir));

    let host_dsos_paths = if let Some(dir) = opt.driver_directory {
        log_info(&format!(
            "Retrieving DSOs from the specified directory: {:?}",
            dir
        ));
        vec![dir]
    } else {
        log_info("Retrieving DSOs from the load path.");
        get_ld_paths()
    };

    let new_env = nvidia_main(&cache_dir, &host_dsos_paths, opt.print_ld_library_path)?;

    if opt.print_ld_library_path {
        return Ok(());
    }

    let Some(nix_binary) = opt.nix_binary else {
        bail!("binary not specified, pass it as an argument!");
    };

    if let Ok(elapsed) = SystemTime::now().duration_since(start_time) {
        log_info(&format!(
            "{:?} seconds elapsed since script start.",
            elapsed
        ));
    }

    for (key, value) in new_env {
        env::set_var(key, value);
    }
    exec_binary(&nix_binary, &opt.args)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_hostdso_json_golden_test() {
        let lp = LibraryPath {
            glx: vec![ResolvedLib {
                name: "dummyglx.so".to_string(),
                dirpath: "/lib".to_string(),
                fullpath: "/lib/dummyglx.so".to_string(),
                last_modification: 1670260550.481498,
                size: 1612,
            }],
            cuda: vec![ResolvedLib {
                name: "dummycuda.so".to_string(),
                dirpath: "/lib".to_string(),
                fullpath: "/lib/dummycuda.so".to_string(),
                last_modification: 2670260550.481498,
                size: 2612,
            }],
            generic: vec![ResolvedLib {
                name: "dummygeneric.so".to_string(),
                dirpath: "/lib".to_string(),
                fullpath: "/lib/dummygeneric.so".to_string(),
                last_modification: 3670260550.481498,
                size: 3612,
            }],
            egl: vec![ResolvedLib {
                name: "dummyegl.so".to_string(),
                dirpath: "/lib".to_string(),
                fullpath: "/lib/dummyegl.so".to_string(),
                last_modification: 4670260550.481498,
                size: 4612,
            }],
            path: "/path/to/lib/dir".to_string(),
        };

        let cdc = CacheDirContent::new(vec![lp]);
        let json = cdc.to_json();

        let golden_cdc = CacheDirContent::from_json(&json).unwrap();
        assert_eq!(cdc, golden_cdc);
        assert_eq!(cdc.to_json(), golden_cdc.to_json());
    }

    #[test]
    fn test_eq_commut_jsons() {
        // Get the current directory of the test file
        let current_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let fixtures_dir = current_dir
            .join("tests")
            .join("fixtures")
            .join("json_permut");

        // Read the test files
        let cdc_json =
            fs::read_to_string(fixtures_dir.join("1.json")).expect("Failed to read 1.json");
        let commut_cdc_json =
            fs::read_to_string(fixtures_dir.join("2.json")).expect("Failed to read 2.json");
        let wrong_cdc_json = fs::read_to_string(fixtures_dir.join("not-equal.json"))
            .expect("Failed to read not-equal.json");

        // Parse the JSON content
        let cdc = CacheDirContent::from_json(&cdc_json).expect("Failed to parse cdc_json");
        let commut_cdc =
            CacheDirContent::from_json(&commut_cdc_json).expect("Failed to parse commut_cdc_json");
        let wrong_cdc =
            CacheDirContent::from_json(&wrong_cdc_json).expect("Failed to parse wrong_cdc_json");

        // Run the assertions
        assert_eq!(cdc, commut_cdc);
        assert_ne!(cdc, wrong_cdc);
        assert_ne!(commut_cdc, wrong_cdc);
    }
}
