use serde_json::{json, Value};
use std::{
    env,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::Mutex,
    thread,
};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use tauri::{Emitter, Manager};

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[derive(Default)]
struct WorkerState {
    process: Mutex<Option<WorkerProcess>>,
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri must be inside desktop_v2")
        .parent()
        .expect("desktop_v2 must be inside project root")
        .to_path_buf()
}

fn emit_worker_line(app: &tauri::AppHandle, line: &str) {
    let payload = serde_json::from_str::<Value>(line)
        .unwrap_or_else(|_| json!({"type": "worker.log", "message": line}));
    let _ = app.emit("worker-event", payload);
}

fn spawn_worker(app: &tauri::AppHandle) -> Result<WorkerProcess, String> {
    let root = project_root();
    let resource_worker = app
        .path()
        .resource_dir()
        .ok()
        .map(|path| path.join("worker").join("课析处理引擎.exe"));

    let (mut command, working_dir) = if resource_worker.as_ref().is_some_and(|path| path.is_file())
    {
        let executable = resource_worker.expect("checked above");
        let directory = executable
            .parent()
            .map(PathBuf::from)
            .ok_or_else(|| "处理引擎资源目录无效".to_string())?;
        (Command::new(executable), directory)
    } else {
        let python = env::var("PYTHON").unwrap_or_else(|_| "python".to_string());
        let mut command = Command::new(&python);
        command.args(["-m", "video_page_detector.desktop_v2_worker"]);
        (command, root)
    };
    command
        .current_dir(working_dir)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        command.creation_flags(0x08000000);
    }

    let mut child = command
        .spawn()
        .map_err(|error| format!("无法启动 Python 处理引擎：{error}"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "无法连接处理引擎输入流".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "无法连接处理引擎输出流".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "无法连接处理引擎错误流".to_string())?;

    let stdout_app = app.clone();
    let stderr_lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let stderr_lines_for_stdout = stderr_lines.clone();

    let stderr_app = app.clone();
    let stderr_handle = thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            match stderr_lines.lock() {
                Ok(mut guard) => {
                    if guard.len() < 500 {
                        guard.push(line.clone());
                    }
                }
                Err(_) => break,
            }
            let _ = stderr_app.emit(
                "worker-event",
                json!({"type": "worker.log", "message": line}),
            );
        }
    });

    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            emit_worker_line(&stdout_app, &line);
        }
        // 等待 stderr 线程结束，确保所有错误输出已被收集
        let _ = stderr_handle.join();
        let stderr_snapshot = match stderr_lines_for_stdout.lock() {
            Ok(guard) => guard.join("\n"),
            Err(_) => String::new(),
        };
        let mut error_msg = "后台处理引擎已停止，请重新启动应用后重试。".to_string();
        if !stderr_snapshot.is_empty() {
            error_msg.push_str("\n引擎错误输出：\n");
            error_msg.push_str(&stderr_snapshot);
        }
        let _ = stdout_app.emit(
            "worker-event",
            json!({
                "type": "worker.exited",
                "error": error_msg
            }),
        );
    });
    Ok(WorkerProcess { child, stdin })
}

#[tauri::command]
fn send_worker_command(
    app: tauri::AppHandle,
    state: tauri::State<WorkerState>,
    command: Value,
) -> Result<(), String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "处理引擎状态锁定失败".to_string())?;
    let must_spawn = match guard.as_mut() {
        Some(process) => process
            .child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_some(),
        None => true,
    };
    if must_spawn {
        *guard = Some(spawn_worker(&app)?);
    }
    let process = guard
        .as_mut()
        .ok_or_else(|| "处理引擎没有启动".to_string())?;
    let serialized =
        serde_json::to_string(&command).map_err(|error| format!("任务参数无法序列化：{error}"))?;
    process
        .stdin
        .write_all(format!("{serialized}\n").as_bytes())
        .and_then(|_| process.stdin.flush())
        .map_err(|error| format!("无法向处理引擎发送任务：{error}"))
}

#[tauri::command]
fn project_output_dir(app: tauri::AppHandle) -> Result<String, String> {
    let output = app
        .path()
        .document_dir()
        .map_err(|error| format!("无法读取文档目录：{error}"))?
        .join("课堂PPT处理结果");
    std::fs::create_dir_all(&output).map_err(|error| format!("无法创建默认结果目录：{error}"))?;
    Ok(output.to_string_lossy().to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(WorkerState::default())
        .invoke_handler(tauri::generate_handler![
            send_worker_command,
            project_output_dir
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            let state = app.state::<WorkerState>();
            if let Ok(mut guard) = state.process.lock() {
                match spawn_worker(&handle) {
                    Ok(worker) => *guard = Some(worker),
                    Err(error) => {
                        let _ = handle.emit(
                            "worker-event",
                            json!({"type": "worker.error", "error": error}),
                        );
                    }
                }
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running kexi desktop");
}
